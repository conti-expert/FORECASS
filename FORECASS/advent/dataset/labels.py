import json
from pathlib import Path

import numpy as np

CLASS_NAMES = ("flat", "construction", "object", "nature", "sky", "human", "vehicle")
PALETTE = np.array([
    [128, 64, 128], [70, 70, 70], [153, 153, 153], [107, 142, 35],
    [70, 130, 180], [220, 20, 60], [0, 0, 142],
], dtype=np.uint8)
SEGMENTATION_MEAN = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def color_to_ids(rgb):
    ids = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        ids[np.all(rgb == color, axis=-1)] = class_id
    return ids


def colorize(ids):
    rgb = np.zeros((*ids.shape, 3), dtype=np.uint8)
    valid = (ids >= 0) & (ids < len(PALETTE))
    rgb[valid] = PALETTE[ids[valid]]
    return rgb


def remap_labels(ids, dataset):
    tables = json.loads(Path(__file__).with_name("mappings.json").read_text())
    mapping = tables[dataset]
    result = np.full(ids.shape, 255, dtype=np.uint8)
    for old, new in mapping.items():
        if new != 7:
            result[ids == int(old)] = new
    return result


def validate_ids(ids):
    values = set(np.unique(ids).tolist())
    if not values.issubset(set(range(7)) | {255}):
        raise ValueError(f"Expected shared class IDs 0..6 or 255, found {sorted(values)}")
