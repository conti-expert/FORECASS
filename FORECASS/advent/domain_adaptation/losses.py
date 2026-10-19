import torch
import torch.nn.functional as F


def weighted_dice_loss_staircase(outputs, targets, error_mask, eps=1e-6, ignore_index=255):
    probs = F.softmax(outputs, dim=1)
    classes = probs.shape[1]
    target = F.one_hot(targets.clamp(0, classes - 1), classes).permute(0, 3, 1, 2).float()
    valid = (targets != ignore_index).float()
    error = error_mask.float().to(outputs.device)
    if error.max() > 1.0:
        error = error / 255.0
    weights = torch.ones_like(error)
    weights[(error > 0.5) & (error <= 0.7)] = 0.5
    weights[(error > 0.7) & (error <= 0.9)] = 0.4
    weights[error > 0.9] = 0.3
    weights = weights * valid
    weights = weights / weights.mean(dim=(1, 2), keepdim=True).clamp_min(eps)
    weights = weights.unsqueeze(1)
    intersection = (probs * target * weights).sum(dim=(2, 3))
    union = ((probs + target) * weights).sum(dim=(2, 3))
    dice = 2 * intersection / (union + eps)
    present_images = valid.sum(dim=(1, 2)) > 0
    if not present_images.any():
        return outputs.sum() * 0.0
    return 1.0 - dice[present_images].mean()


def cosine_pairwise(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=1)
    x = x.transpose(1, 2).contiguous()
    return torch.bmm(x, x.transpose(1, 2))

def compute_consistency_loss(student_logits: torch.Tensor,
                             teacher_logits: torch.Tensor,
                             n_pair: int = 512,
                             lambda_sc: float = 1.0) -> torch.Tensor:
    stu_flat = student_logits.flatten(2)
    with torch.no_grad():
        tea_flat = teacher_logits.flatten(2).detach()

    num_pix = stu_flat.shape[-1]
    idx = torch.randperm(num_pix, device=stu_flat.device)[:n_pair]
    stu_sel = stu_flat[:, :, idx]
    tea_sel = tea_flat[:, :, idx]
    loss = F.mse_loss(cosine_pairwise(stu_sel),
                      cosine_pairwise(tea_sel))
    return lambda_sc * loss

def compute_distillation_loss(student_logits, teacher_logits, temperature):
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
    return loss
