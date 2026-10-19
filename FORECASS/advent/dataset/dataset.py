import csv
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from advent.dataset.labels import SEGMENTATION_MEAN, validate_ids


def read_manifest(path, require_labels=False):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "image" not in reader.fieldnames:
            raise ValueError(f"{path}: CSV must have an image column")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    seen = set()
    for row in rows:
        name = row["image"].replace("\\", "/")
        p = PurePosixPath(name)
        if not name or p.is_absolute() or ".." in p.parts or ":" in name:
            raise ValueError(f"Expected a relative image path: {name}")
        row["image"] = str(p)
        label = row.get("label", "").replace("\\", "/")
        lp = PurePosixPath(label)
        if label and (lp.is_absolute() or ".." in lp.parts or ":" in label):
            raise ValueError(f"Expected a relative label path: {label}")
        row["label"] = label
        key = cache_name(name)
        if key in seen:
            raise ValueError(f"Duplicate output name in {path}: {key}")
        seen.add(key)
        if require_labels and not label:
            raise ValueError(f"Missing evaluation label for {name}")
    return rows


def cache_name(name):
    return str(PurePosixPath(name).with_suffix(".png"))


class SegmentationDataset:
    def __init__(self, root, manifest, size=(640, 320), labels=False, cache=None):
        self.root = Path(root)
        self.rows = read_manifest(manifest, require_labels=labels)
        self.size = tuple(size)
        self.labels = labels
        self.cache = Path(cache) if cache else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image"]) as source:
            rgb = source.convert("RGB").resize(self.size, Image.Resampling.BICUBIC)
        rgb = np.array(rgb, dtype=np.float32)
        image = (rgb[..., ::-1] - SEGMENTATION_MEAN).transpose(2, 0, 1).copy()
        name = cache_name(row["image"])
        if self.cache:
            pseudo = self._read_ids(self.cache / "pseudo" / name)
            with Image.open(self.cache / "probability" / name) as source:
                error = np.array(source.convert("L").resize(self.size, Image.Resampling.NEAREST))
            return image, pseudo.astype(np.int64), error.astype(np.float32) / 255.0, name
        if self.labels:
            target = self._read_ids(self.root / row["label"])
            return image, target.astype(np.int64), name
        return image, name

    def _read_ids(self, path):
        with Image.open(path) as source:
            if source.mode not in ("L", "P", "I", "I;16"):
                raise ValueError(f"Expected an indexed shared-class mask: {path}")
            result = np.array(source.resize(self.size, Image.Resampling.NEAREST))
        validate_ids(result)
        return result
