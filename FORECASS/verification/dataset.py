import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from advent.dataset.labels import color_to_ids, validate_ids
from verification.inference import make_input


class VerificationDataset:
    def __init__(self, root, manifest, augment=False, size=(640, 320), mask_format="ids",
                 color_jitter_strength=0.25, color_jitter_probability=0.2):
        self.root = Path(root)
        self.augment = augment
        self.size = tuple(size)
        self.mask_format = mask_format
        self.color_jitter_probability = color_jitter_probability
        with open(manifest, newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not {"image", "prediction", "error"}.issubset(reader.fieldnames or []):
                raise ValueError("Verification CSV needs image,prediction,error columns")
            self.rows = list(reader)
        if not self.rows:
            raise ValueError(f"Empty verification manifest: {manifest}")
        strength = color_jitter_strength
        self.jitter = T.ColorJitter(brightness=strength, contrast=strength, saturation=strength, hue=strength)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image"]) as source:
            rgb = source.convert("RGB")
        with Image.open(self.root / row["prediction"]) as source:
            ids = color_to_ids(np.array(source.convert("RGB"))) if self.mask_format == "color" else np.array(source)
        validate_ids(ids)
        mask = Image.fromarray(ids.astype(np.uint8))
        with Image.open(self.root / row["error"]) as source:
            error = source.convert("L")
        if self.augment:
            if random.random() < 0.5:
                rgb, mask, error = [TF.hflip(x) for x in (rgb, mask, error)]
            scale = random.uniform(0.5, 2.0)
            width, height = [int(x * scale) for x in self.size]
            rgb = rgb.resize((width, height), Image.Resampling.BILINEAR)
            mask = mask.resize((width, height), Image.Resampling.NEAREST)
            error = error.resize((width, height), Image.Resampling.NEAREST)
            padding = (0, 0, max(0, self.size[0] - width), max(0, self.size[1] - height))
            rgb = ImageOps.expand(rgb, padding, fill=0)
            mask = ImageOps.expand(mask, padding, fill=255)
            error = ImageOps.expand(error, padding, fill=0)
            left = random.randint(0, rgb.width - self.size[0])
            top = random.randint(0, rgb.height - self.size[1])
            box = (left, top, left + self.size[0], top + self.size[1])
            rgb, mask, error = [x.crop(box) for x in (rgb, mask, error)]
            if random.random() < self.color_jitter_probability:
                rgb = self.jitter(rgb)
            if random.random() < 0.5:
                rgb = rgb.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 1.5)))
        else:
            rgb = rgb.resize(self.size, Image.Resampling.BILINEAR)
            mask = mask.resize(self.size, Image.Resampling.NEAREST)
            error = error.resize(self.size, Image.Resampling.NEAREST)
        ids = np.array(mask)
        x = make_input(np.array(rgb), ids)
        y = torch.from_numpy((np.array(error) > 0).astype(np.float32))[None]
        valid = torch.from_numpy(ids != 255)[None]
        return x, y, valid
