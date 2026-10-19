import torch
import torch.nn.functional as F


def refiner_loss(logits, targets, valid, alpha=0.25, gamma=2.0, eps=1e-6):
    probability = logits.sigmoid()
    valid = valid.to(logits.dtype)
    log_positive = F.logsigmoid(logits)
    log_negative = F.logsigmoid(-logits)
    focal = -(alpha * (1 - probability).pow(gamma) * targets * log_positive
              + (1 - alpha) * probability.pow(gamma) * (1 - targets) * log_negative)
    focal = (focal * valid).flatten(1).sum(1)
    intersection = (probability * targets * valid).flatten(1).sum(1)
    denominator = ((probability.square() + targets.square()) * valid).flatten(1).sum(1)
    dice = 1 - 2 * intersection / (denominator + eps)
    present = valid.flatten(1).sum(1) > 0
    if not present.any():
        return logits.sum() * 0.0
    return (focal[present] + 0.5 * dice[present]).mean()
