import kornia
import numpy as np
import torch

from advent.dataset.labels import SEGMENTATION_MEAN


def strong_augment(images, strength=0.25, probability=0.2, blur=True):
    mean = torch.as_tensor(SEGMENTATION_MEAN, device=images.device).view(1, 3, 1, 1)
    rgb = ((images + mean) / 255.0).flip(1)
    jitter = kornia.augmentation.ColorJitter(
        brightness=strength, contrast=strength, saturation=strength, hue=strength,
        p=probability, same_on_batch=False,
    )
    rgb = jitter(rgb)
    if blur:
        height, width = rgb.shape[-2:]
        kernel = tuple(max(1, int(np.ceil(0.1 * size)) // 2 * 2 + 1) for size in (height, width))
        rgb = kornia.augmentation.RandomGaussianBlur(
            kernel_size=kernel, sigma=(0.15, 1.15), p=0.5, same_on_batch=False,
        )(rgb)
    return rgb.flip(1) * 255.0 - mean
