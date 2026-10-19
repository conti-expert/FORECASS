import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from advent.model.deeplabv2 import get_deeplab_v2
from advent.utils.checkpoints import load_checkpoint, load_model, restore_rng, save_checkpoint, seed_worker, set_seed
from advent.utils.config import load_config, save_config
from advent.dataset.dataset import SegmentationDataset
from advent.scripts.test import evaluate
from advent.domain_adaptation.losses import compute_consistency_loss, compute_distillation_loss, weighted_dice_loss_staircase
from advent.scripts.prepare import check_cache, prepare_cache
from advent.model.transforms import strong_augment


def main():
    parser = argparse.ArgumentParser(description="Train ForeCASS on one target domain")
    parser.add_argument("--cfg", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--init", help="Initialize a new domain; reset optimizer and iteration")
    group.add_argument("--resume", help="Resume an interrupted run on the same domain")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, help="Updates for this domain; not specified in the paper")
    parser.add_argument("--refresh-every", type=int, help="Pseudo-label and error-map refresh interval")
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    train = cfg["training"]
    if args.iterations is not None:
        train["iterations"] = args.iterations
        train["lr_max_iterations"] = args.iterations
    if args.refresh_every is not None:
        train["pseudo_refresh_every"] = args.refresh_every
    for key in ("iterations", "lr_max_iterations", "pseudo_refresh_every"):
        if not isinstance(train[key], int) or train[key] <= 0:
            raise ValueError(f"Set a positive {key}; this schedule parameter is not reported in the paper")
    set_seed(cfg["seed"])
    device = torch.device(args.device)
    cache = check_cache(cfg)
    cache_metadata = json.loads((cache / "manifest.json").read_text())
    if args.init and cache_metadata.get("initialization") != str(Path(args.init).resolve()):
        raise ValueError("Prepare the cache with the same --init model used for training")
    output = Path(cfg["output_dir"])
    if not args.resume and (output / "latest.pth").exists():
        raise FileExistsError(f"{output} already contains a run; use --resume or another output_dir")
    ds = cfg["dataset"]
    dataset = SegmentationDataset(ds["root"], ds["train_manifest"], ds["size"], cache=cache)
    loader = DataLoader(dataset, batch_size=train["batch_size"], shuffle=True,
                        num_workers=cfg["workers"], pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
    validation = None
    if ds.get("val_manifest"):
        validation = DataLoader(SegmentationDataset(ds["root"], ds["val_manifest"], ds["size"], labels=True),
                                batch_size=1, num_workers=cfg["workers"], shuffle=False)
    model = get_deeplab_v2(num_classes=7, multi_level=True).to(device)
    checkpoint = load_checkpoint(args.resume or args.init)
    load_model(model, checkpoint)
    model.train()
    teacher = copy.deepcopy(model)
    load_model(teacher, checkpoint, teacher=True)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    optimizer = torch.optim.SGD(model.optim_parameters(train["learning_rate"]),
                                lr=train["learning_rate"], momentum=train["momentum"],
                                weight_decay=train["weight_decay"])
    start, best = 0, -1.0
    if args.resume:
        if checkpoint.get("config") != cfg:
            raise ValueError("Resume requires the same configuration as the saved run")
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start = checkpoint["iter"]
        best = checkpoint["best_mIoU"]
        restore_rng(checkpoint)
    save_config(cfg, output / "config.yaml")
    if not 0 <= start < train["iterations"] <= train["lr_max_iterations"]:
        raise ValueError("Expected 0 <= start < iterations <= lr_max_iterations")
    iterator = iter(loader)
    progress = tqdm(range(start, train["iterations"]), desc="ForeCASS")
    for iteration in progress:
        lr = train["learning_rate"] * (1 - iteration / train["lr_max_iterations"]) ** train["lr_power"]
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[1]["lr"] = 10 * lr
        try:
            images, labels, errors, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, labels, errors, _ = next(iterator)
        images, labels, errors = images.to(device), labels.to(device), errors.to(device)
        teacher_images = images.clone()
        if train["strong_transform"]:
            images = strong_augment(images, train["color_jitter_strength"],
                                    train["color_jitter_probability"], train["blur"])
        optimizer.zero_grad()
        prediction = model(images)[1]
        prediction = F.interpolate(prediction, size=labels.shape[-2:], mode="bilinear", align_corners=True)
        with torch.no_grad():
            teacher_prediction = teacher(teacher_images)[1]
            teacher_prediction = F.interpolate(teacher_prediction, size=labels.shape[-2:], mode="bilinear", align_corners=True)
        dice = weighted_dice_loss_staircase(prediction, labels.long(), errors.float())
        distillation = compute_distillation_loss(prediction, teacher_prediction, train["distill_temperature"])
        consistency = compute_consistency_loss(prediction, teacher_prediction, n_pair=train["consistency_pairs"])
        loss = dice + train["distill_weight"] * distillation + train["consistency_weight"] * consistency
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for teacher_parameter, student_parameter in zip(teacher.parameters(), model.parameters()):
                teacher_parameter.mul_(train["ema_alpha"]).add_(student_parameter, alpha=1 - train["ema_alpha"])
            for teacher_buffer, student_buffer in zip(teacher.buffers(), model.buffers()):
                if teacher_buffer.is_floating_point():
                    teacher_buffer.mul_(train["ema_alpha"]).add_(student_buffer, alpha=1 - train["ema_alpha"])
                else:
                    teacher_buffer.copy_(student_buffer)
        completed = iteration + 1
        values = {"dice": dice.item(), "distillation": distillation.item(),
                  "consistency": consistency.item(), "total": loss.item()}
        progress.set_postfix(loss=f"{loss.item():.4f}")
        if completed % train["log_every"] == 0:
            with (output / "training.jsonl").open("a") as stream:
                stream.write(json.dumps({"iter": completed, "lr": lr, **values}) + "\n")
        if validation is not None and completed % train["evaluate_every"] == 0:
            stats = evaluate(teacher, validation, device)
            print(f"Iteration {completed}: mIoU {stats['miou']:.2f}", flush=True)
            with (output / "evaluation.jsonl").open("a") as stream:
                stream.write(json.dumps({"iter": completed, **stats}) + "\n")
            if stats["miou"] > best:
                best = stats["miou"]
                save_checkpoint(output / "best.pth", model, teacher, optimizer, completed, best, cfg)
        if completed < train["iterations"] and completed % train["pseudo_refresh_every"] == 0:
            iterator = None
            prepare_cache(teacher, cfg, cache_metadata["verification_model"], device,
                          cache_metadata.get("initialization"))
            teacher.eval()
            iterator = iter(loader)
        if completed % train["save_every"] == 0 or completed == train["iterations"]:
            save_checkpoint(output / "latest.pth", model, teacher, optimizer, completed, best, cfg)


if __name__ == "__main__":
    main()
