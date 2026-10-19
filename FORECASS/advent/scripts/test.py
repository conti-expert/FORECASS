import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from advent.model.deeplabv2 import get_deeplab_v2
from advent.utils.checkpoints import load_checkpoint, load_model
from advent.utils.config import load_config
from advent.dataset.dataset import SegmentationDataset
from advent.dataset.labels import CLASS_NAMES, colorize


@torch.no_grad()
def evaluate(model, loader, device, output_dir=None):
    from PIL import Image

    was_training = model.training
    model.eval()
    hist = np.zeros((7, 7), dtype=np.int64)
    try:
        for images, labels, names in tqdm(loader, desc="Evaluation"):
            logits = model(images.to(device))[1]
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=True)
            predicted = logits.argmax(1).cpu().numpy()
            for prediction, target, name in zip(predicted, labels.numpy(), names):
                valid = (target >= 0) & (target < 7)
                hist += np.bincount(7 * target[valid] + prediction[valid], minlength=49).reshape(7, 7)
                if output_dir:
                    path = Path(output_dir) / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(colorize(prediction)).save(path)
    finally:
        model.train(was_training)
    union = hist.sum(0) + hist.sum(1) - hist.diagonal()
    ious = np.divide(hist.diagonal(), union, out=np.full(7, np.nan), where=union > 0) * 100
    if not np.isfinite(ious).any():
        raise ValueError("Evaluation contains no valid shared-class pixels")
    return {"miou": float(np.nanmean(ious)), "class_iou": {
        name: float(iou) if np.isfinite(iou) else None for name, iou in zip(CLASS_NAMES, ious)
    }, "confusion_matrix": hist.tolist()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a ForeCASS segmentation model")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--student", action="store_true", help="Evaluate student weights instead of EMA weights")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--predictions")
    parser.add_argument("--output", default="evaluation.json")
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    ds = cfg["dataset"]
    dataset = SegmentationDataset(ds["root"], ds["val_manifest"], ds["size"], labels=True)
    loader = DataLoader(dataset, batch_size=1, num_workers=cfg["workers"], shuffle=False)
    device = torch.device(args.device)
    model = get_deeplab_v2(num_classes=7, multi_level=True).to(device)
    load_model(model, load_checkpoint(args.model), teacher=not args.student)
    result = evaluate(model, loader, device, args.predictions)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
