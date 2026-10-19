import random

import numpy as np
import torch


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model(model, checkpoint, teacher=False):
    key = "teacher_state" if teacher and "teacher_state" in checkpoint else "model_state"
    state = checkpoint.get(key, checkpoint)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(path, model, teacher, optimizer, iteration, best_miou, cfg):
    torch.save({
        "model_state": model.state_dict(), "teacher_state": teacher.state_dict(),
        "optimizer_state": optimizer.state_dict(), "iter": iteration,
        "best_mIoU": best_miou, "config": cfg,
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)


def restore_rng(checkpoint):
    if "python_rng" in checkpoint:
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
