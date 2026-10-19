import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from advent.model.deeplabv2 import get_deeplab_v2
from advent.utils.checkpoints import load_checkpoint, load_model, save_checkpoint, seed_worker, set_seed
from advent.utils.config import load_config, save_config
from advent.dataset.dataset import SegmentationDataset
from advent.scripts.test import evaluate


def main():
    parser = argparse.ArgumentParser(description="Supervised source pretraining for ForeCASS")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--init", help="Optional full seven-class DeepLabV2 initialization")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    cfg = load_config(args.cfg)
    train, ds = cfg["training"], cfg["dataset"]
    train["iterations"] = args.iterations
    output = Path(cfg["output_dir"])
    if (output / "latest.pth").exists():
        raise FileExistsError(f"Use a new output_dir: {output}")
    output.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output / "config.yaml")
    set_seed(cfg["seed"])
    device = torch.device(args.device)
    model = get_deeplab_v2(num_classes=7, multi_level=True).to(device)
    if args.init:
        load_model(model, load_checkpoint(args.init))
    model.train()
    dataset = SegmentationDataset(ds["root"], ds["train_manifest"], ds["size"], labels=True)
    loader = DataLoader(dataset, batch_size=train["batch_size"], shuffle=True,
                        num_workers=cfg["workers"], worker_init_fn=seed_worker)
    validation = None
    if ds.get("val_manifest"):
        validation = DataLoader(SegmentationDataset(ds["root"], ds["val_manifest"], ds["size"], labels=True),
                                batch_size=1, num_workers=cfg["workers"], shuffle=False)
    optimizer = torch.optim.SGD(model.optim_parameters(train["learning_rate"]),
                                lr=train["learning_rate"], momentum=train["momentum"],
                                weight_decay=train["weight_decay"])
    iterator = iter(loader)
    best = -1.0
    progress = tqdm(range(args.iterations), desc="Source pretraining")
    for iteration in progress:
        try:
            images, labels, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, labels, _ = next(iterator)
        images, labels = images.to(device), labels.to(device)
        if not (labels != 255).any():
            raise ValueError("Source batch contains no valid labels")
        lr = train["learning_rate"] * (1 - iteration / args.iterations) ** train["lr_power"]
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[1]["lr"] = 10 * lr
        prediction = model(images)[1]
        prediction = F.interpolate(prediction, size=labels.shape[-2:], mode="bilinear", align_corners=True)
        loss = F.cross_entropy(prediction, labels, ignore_index=255)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        completed = iteration + 1
        progress.set_postfix(loss=f"{loss.item():.4f}")
        if validation is not None and completed % train["evaluate_every"] == 0:
            result = evaluate(model, validation, device)
            print(f"Iteration {completed}: mIoU {result['miou']:.2f}", flush=True)
            if result["miou"] > best:
                best = result["miou"]
                save_checkpoint(output / "best.pth", model, model, optimizer, completed, best, cfg)
        if completed % train["save_every"] == 0 or completed == args.iterations:
            save_checkpoint(output / "latest.pth", model, model, optimizer, completed, best, cfg)


if __name__ == "__main__":
    main()
