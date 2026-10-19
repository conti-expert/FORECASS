from pathlib import Path

import yaml


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping")
    required = {"dataset", "training", "output_dir"}
    if not required.issubset(cfg):
        raise ValueError(f"Missing configuration fields: {required - cfg.keys()}")
    return cfg


def save_config(cfg, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
