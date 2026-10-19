import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from advent.utils.checkpoints import load_checkpoint, load_model, seed_worker, set_seed
from verification.dataset import VerificationDataset
from verification.model import get_error_detector
from verification.losses import refiner_loss


def run_epoch(model, loader, device, alpha, gamma, optimizer=None):
    model.train(optimizer is not None)
    total_loss = count = 0
    tp = fp = fn = 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y, valid in tqdm(loader, leave=False):
            x, y, valid = x.to(device), y.to(device), valid.to(device)
            if not valid.any():
                continue
            logits = model(x)
            loss = refiner_loss(logits, y, valid, alpha=alpha, gamma=gamma)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            count += x.size(0)
            predicted = logits.sigmoid()[valid] > 0.5
            truth = y[valid] > 0.5
            tp += (predicted & truth).sum().item()
            fp += (predicted & ~truth).sum().item()
            fn += (~predicted & truth).sum().item()
    if not count:
        raise ValueError("No valid verification pixels in this split")
    return {"loss": total_loss / count, "error_iou": tp / max(tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def main():
    parser = argparse.ArgumentParser(description="Train the ForeCASS verification model")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    with open(args.cfg, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    set_seed(cfg["seed"])
    device = torch.device(args.device)
    options = dict(root=cfg["root"], size=cfg["size"], mask_format=cfg["mask_format"],
                   color_jitter_strength=cfg["color_jitter_strength"],
                   color_jitter_probability=cfg["color_jitter_probability"])
    training = VerificationDataset(manifest=cfg["train_manifest"], augment=True, **options)
    validation = VerificationDataset(manifest=cfg["val_manifest"], augment=False, **options)
    train_images = {r["image"] for r in training.rows}
    if train_images & {r["image"] for r in validation.rows}:
        raise ValueError("Verification training and validation contain the same image paths")
    loader_options = dict(batch_size=cfg["batch_size"], num_workers=cfg["workers"], worker_init_fn=seed_worker)
    train_loader = DataLoader(training, shuffle=True, **loader_options)
    val_loader = DataLoader(validation, shuffle=False, **loader_options)
    model = get_error_detector(cfg["backbone"], None if args.resume else cfg["encoder_weights"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    start, best = 1, -1.0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        load_model(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start = checkpoint["epoch"] + 1
        best = checkpoint.get("best_error_iou", checkpoint.get("best_miou", -1.0))
    for epoch in range(start, cfg["epochs"] + 1):
        train_stats = run_epoch(model, train_loader, device, cfg["focal_alpha"], cfg["focal_gamma"], optimizer)
        val_stats = run_epoch(model, val_loader, device, cfg["focal_alpha"], cfg["focal_gamma"])
        scheduler.step(val_stats["error_iou"])
        improved = val_stats["error_iou"] > best
        best = max(best, val_stats["error_iou"])
        checkpoint = {"epoch": epoch, "model_state": model.state_dict(),
                      "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                      "best_error_iou": best, "config": cfg}
        torch.save(checkpoint, output / "latest.pth")
        if improved:
            torch.save(checkpoint, output / "best.pth")
        print(f"Epoch {epoch}: train={train_stats}, validation={val_stats}", flush=True)


if __name__ == "__main__":
    main()
