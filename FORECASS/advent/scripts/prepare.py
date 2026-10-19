import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from advent.model.deeplabv2 import get_deeplab_v2
from advent.utils.checkpoints import load_checkpoint, load_model
from advent.utils.config import load_config
from advent.dataset.dataset import SegmentationDataset
from verification.inference import generate_maps
from verification.model import get_error_detector


def prepare_cache(teacher, cfg, verification_model, device, initialization=None):
    ds = cfg["dataset"]
    cache = Path(cfg["output_dir"]) / "cache"
    marker = cache / "manifest.json"
    marker.unlink(missing_ok=True)
    dataset = SegmentationDataset(ds["root"], ds["train_manifest"], ds["size"])
    loader = DataLoader(dataset, batch_size=1, num_workers=cfg["workers"], shuffle=False)
    teacher.eval()
    with torch.no_grad():
        for images, names in tqdm(loader, desc="Pseudo-labels"):
            logits = teacher(images.to(device))[1]
            logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=True)
            ids = logits.softmax(1).argmax(1)[0].cpu().numpy().astype(np.uint8)
            path = cache / "pseudo" / names[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(ids).save(path)
    verifier = get_error_detector(cfg["verification_backbone"]).to(device)
    load_model(verifier, load_checkpoint(verification_model))
    generate_maps(verifier, ds["root"], dataset.rows, cache / "pseudo",
                  cache / "probability", ds["size"], device)
    marker.write_text(json.dumps({"dataset": ds, "rows": dataset.rows,
                                 "verification_model": str(Path(verification_model).resolve()),
                                 "initialization": initialization}, indent=2) + "\n")
    return cache


def check_cache(cfg):
    cache = Path(cfg["output_dir"]) / "cache"
    marker = cache / "manifest.json"
    if not marker.exists():
        raise ValueError("Prepared cache is missing or incomplete; run advent.scripts.prepare first")
    metadata = json.loads(marker.read_text())
    dataset = SegmentationDataset(cfg["dataset"]["root"], cfg["dataset"]["train_manifest"],
                                  cfg["dataset"]["size"], cache=cache)
    if metadata["dataset"] != cfg["dataset"] or metadata["rows"] != dataset.rows:
        raise ValueError("Cache belongs to a different dataset configuration or manifest")
    from advent.dataset.dataset import cache_name
    for row in dataset.rows:
        name = cache_name(row["image"])
        for kind in ("pseudo", "probability"):
            if not (cache / kind / name).is_file():
                raise FileNotFoundError(cache / kind / name)
    return cache


def main():
    parser = argparse.ArgumentParser(description="Generate target pseudo-labels and verification maps")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--verification-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    device = torch.device(args.device)
    teacher = get_deeplab_v2(num_classes=7, multi_level=True).to(device)
    load_model(teacher, load_checkpoint(args.init), teacher=True)
    print(prepare_cache(teacher, cfg, args.verification_model, device, str(Path(args.init).resolve())))


if __name__ == "__main__":
    main()
