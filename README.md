# FORECASS: Forgetting-Resilient Continuous Source-Free UDA for Semantic Segmentation

[Paper](https://doi.org/10.1016/j.eswa.2026.133534) · Baqar Abbas, Abderrazak Chahi, Yassine Ruichek

## Abstract

Semantic segmentation is fundamental for autonomous driving, requiring dense scene understanding across diverse and continuously changing environments. Real-world deployment must overcome two major challenges: (1) dynamic environmental shifts due to weather, lighting, and geography, and (2) the inability to retain labeled source data for continual adaptation. To address these issues, we propose a novel source-free Adaptive and Forgetting-Resilient Continual Unsupervised Domain Adaptation for Semantic Segmentation (FORECASS). We introduce a teacher–student based framework, using EMA (Exponential Moving Average) updating technique, to produce stable pseudo-labels during continual adaptation. Central to our method is a refiner-based error estimation model that predicts pixel-wise pseudo-label reliability during adaptation. By leveraging the error map, the model selectively focuses learning on uncertain and challenging regions, playing a critical role in mitigating catastrophic forgetting. Complementarily, a structure-aware consistency mechanism enforces semantic coherence across views, further enhancing stability during sequential adaptation. Lightweight knowledge distillation is also incorporated to smooth alignment between the pseudo-label generator and the adaptation model. We validate our framework on the challenging continual adaptation sequence GTA → Cityscapes → IDD → Mapillary, achieving state-of-the-art results over both source-dependent and source-free baselines.

## Installation

Use Python 3.10 with **PyTorch 2.1.0**, the PyTorch version reported in the paper. For CUDA 11.8:

```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Run all commands from the repository root.

## Dataset preparation

Set dataset roots and manifest paths in the corresponding files under `configs/`.

Target training CSVs need an `image` column. Evaluation CSVs also need a `label` column. Paths inside each CSV are relative to the configured dataset root:

```csv
image,label
leftImg8bit/val/frankfurt/example_leftImg8bit.png,labels7/val/frankfurt/example_gtFine_labelIds.png
```

Prepared labels use IDs **0–6** for flat, construction, object, nature, sky, human, and vehicle; **255** is ignored.

Use `prepare_data.py` to convert labels and create manifests:

```bash
python prepare_data.py labels --help
python prepare_data.py manifest --help
```

Label conversion supports `cityscapes`, `idd`, `mapillary`, `gta_trainids` for 19-class train IDs, and `gta_labelids` for original label IDs. The revised mapping is the default; use `--mapping sota` for the previous evaluation mapping.

For native Mapillary JPEGs, use `--image-suffix .jpg --label-suffix .png` when creating manifests.

## Training

### Source model

Set the paths in `configs/source.yaml`, then run:

```bash
python -m advent.scripts.train_source --cfg configs/source.yaml --iterations SOURCE_UPDATES
```

### Verification model

The verification model is MA-Net with a ResNet-101 encoder and four input channels: RGB plus the predicted class-ID map.

Its source train/validation CSVs use `image,prediction,error` columns. Use `prepare_data.py errors` to create binary error targets from source predictions and ground truth, and `prepare_data.py split` to create a train/validation split.

```bash
python -m verification.train --cfg configs/verification.yaml
```

### Continual adaptation

Generate target pseudo-labels and predicted error maps:

```bash
python -m advent.scripts.prepare --cfg configs/cityscapes.yaml --init /path/to/source.pth --verification-model /path/to/refiner.pth
```

Start adaptation:

```bash
python -m advent.scripts.train --cfg configs/cityscapes.yaml --init /path/to/source.pth --iterations DOMAIN_UPDATES --refresh-every REFRESH_INTERVAL
```

Replace `SOURCE_UPDATES`, `DOMAIN_UPDATES`, and `REFRESH_INTERVAL` with positive integers. These schedules are not specified in the paper and are left for the user to choose.

Pseudo-labels and error maps are refreshed automatically during adaptation. The verification model stays frozen, while the segmentation backbone remains trainable.

Continue with:

- `configs/idd.yaml`, initialized from `runs/cityscapes/latest.pth`.
- `configs/mapillary.yaml`, initialized from `runs/idd/latest.pth`.

Run preparation before training each domain. Use `--device cuda:0` to select a GPU.

### Checkpoints and resume

For segmentation training, `latest.pth` is saved every `save_every` updates and at completion. `best.pth` is saved when validation mIoU improves, if validation is configured.

For verification training, `latest.pth` is saved after every epoch, and `best.pth` when validation error-map IoU improves.

Resume target adaptation with:

```bash
python -m advent.scripts.train --cfg runs/cityscapes/config.yaml --resume runs/cityscapes/latest.pth
```

## Model evaluation

```bash
python -m advent.scripts.test --cfg configs/cityscapes.yaml --model runs/mapillary/latest.pth --output results/cityscapes.json
```

Repeat with the IDD and Mapillary configs to evaluate the same model on all three domains.

Evaluation uses EMA teacher weights by default; `--student` selects the student. Target ground truth is used only for evaluation.

## Offline evaluation of saved predictions

`offline_validation.py` computes mIoU directly from saved segmentation predictions and ground-truth maps.

[Download predictions and ground truth](https://utbm-my.sharepoint.com/:f:/g/personal/baqar_abbas_utbm_fr/EorrpuM9biZGhneKXxDDdccBHZsnkvyqHaMHYWgVwJcSYw?e=fFAUGL)

The download contains:

- Predicted semantic segmentation maps generated by FORECASS.
- Ground-truth maps for the SOTA 7-class mapping.
- Ground-truth maps for the FORECASS revised 7-class mapping.

Filenames follow:

- Predictions: `<id>_leftImg8bit.png`
- Ground truth: `<id>_gtFine_labelIds.png`

`<id>` may be `000290`, `frankfurt_000000_000294`, or `mapillary_000000`. The script supports flat folders and Cityscapes-style subfolders.

Download and unzip the folders, then edit the paths in `offline_validation.py`:

```python
PRED_ROOT = r"PATH_TO_PRED_FOLDER"
GT_ROOT = r"PATH_TO_GT_FOLDER"
```

Run:

```bash
python offline_validation.py
```

## Acknowledgements

The code is adapted from [ADVENT](https://github.com/valeoai/ADVENT). The verification model uses [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch). Upstream attribution is retained in `LICENSE` and `NOTICE`.
