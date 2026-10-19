import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from advent.utils.checkpoints import load_checkpoint, load_model
from advent.dataset.dataset import cache_name, read_manifest
from advent.dataset.labels import IMAGENET_MEAN, IMAGENET_STD, color_to_ids, validate_ids
from verification.model import get_error_detector


def make_input(rgb, ids):
    validate_ids(ids)
    rgb = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    safe_ids = np.where(ids == 255, 0, ids)
    semantic = torch.from_numpy(safe_ids.astype(np.float32))[None]
    return torch.cat([torch.from_numpy(rgb).permute(2, 0, 1), semantic], dim=0)


def generate_maps(model, root, rows, pseudo_dir, output_dir, size, device, color_masks=False):
    model.eval()
    root, pseudo_dir, output_dir = map(Path, (root, pseudo_dir, output_dir))
    with torch.no_grad():
        for row in tqdm(rows, desc="Verification maps"):
            name = cache_name(row["image"])
            with Image.open(root / row["image"]) as source:
                rgb = np.array(source.convert("RGB").resize(tuple(size), Image.Resampling.BILINEAR))
            with Image.open(pseudo_dir / name) as source:
                if color_masks:
                    ids = color_to_ids(np.array(source.convert("RGB").resize(tuple(size), Image.Resampling.NEAREST)))
                else:
                    if source.mode not in ("L", "P", "I", "I;16"):
                        raise ValueError("RGB masks require --color-masks")
                    ids = np.array(source.resize(tuple(size), Image.Resampling.NEAREST))
            x = make_input(rgb, ids).unsqueeze(0).to(device)
            probability = model(x).sigmoid()[0, 0].cpu().numpy()
            path = output_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((probability * 255).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser(description="Generate verification probabilities for segmentation masks")
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backbone", default="resnet101")
    parser.add_argument("--size", nargs=2, type=int, default=[640, 320], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--color-masks", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = get_error_detector(args.backbone).to(device)
    load_model(model, load_checkpoint(args.model))
    generate_maps(model, args.root, read_manifest(args.manifest), args.masks,
                  args.output, args.size, device, args.color_masks)


if __name__ == "__main__":
    main()
