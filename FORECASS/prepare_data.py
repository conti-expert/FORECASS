import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from advent.dataset.dataset import cache_name, read_manifest
from advent.dataset.labels import color_to_ids, colorize, remap_labels, validate_ids

def labels(argv):
    parser = argparse.ArgumentParser(description="Convert annotations to the seven shared classes")
    parser.add_argument("--dataset", choices=["cityscapes", "idd", "mapillary", "gta_trainids", "gta_labelids"], required=True)
    parser.add_argument("--mapping", choices=["revised", "sota"], default="revised")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pattern", default="*.png", help="Recursive filename pattern")
    parser.add_argument("--mapillary-config", help="Original Mapillary Vistas v1.2 config.json for RGB annotations")
    parser.add_argument("--color-output", help="Optional separate directory for color previews")
    args = parser.parse_args(argv)
    source, output = Path(args.input).resolve(), Path(args.output).resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be separate from the annotation input tree")
    previews = Path(args.color_output).resolve() if args.color_output else None
    if previews and (previews == source or source in previews.parents or previews == output):
        raise ValueError("Color previews need a separate directory")
    colors = None
    if args.mapillary_config:
        config = json.loads(Path(args.mapillary_config).read_text())
        colors = {tuple(label["color"]): i for i, label in enumerate(config["labels"])}
    files = sorted(source.rglob(args.pattern))
    if not files:
        raise ValueError(f"No labels found in {source}")
    for path in files:
        with Image.open(path) as image:
            if image.mode in ("RGB", "RGBA"):
                if args.dataset != "mapillary" or colors is None:
                    raise ValueError(f"{path}: expected indexed annotations; Mapillary RGB requires --mapillary-config")
                rgb = np.array(image.convert("RGB"))
                ids = np.full(rgb.shape[:2], -1, dtype=np.int32)
                for color, class_id in colors.items():
                    ids[np.all(rgb == color, axis=-1)] = class_id
            else:
                ids = np.array(image)
        dataset = "cityscapes" if args.dataset == "gta_labelids" else args.dataset
        if args.mapping == "sota" and dataset in ("idd", "mapillary"):
            dataset += "_sota"
        labels = remap_labels(ids, dataset)
        relative = path.relative_to(source)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(labels).save(destination)
        if previews:
            preview = previews / relative
            preview.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(colorize(labels)).save(preview)
    print(f"Converted {len(files)} annotations to {output}")

def manifest(argv):
    parser = argparse.ArgumentParser(description="List images and optional shared-class labels")
    parser.add_argument("--root", required=True)
    parser.add_argument("--images", required=True, help="Image directory, relative to root")
    parser.add_argument("--labels", help="Prepared label directory, relative to root")
    parser.add_argument("--image-suffix", default="_leftImg8bit.png")
    parser.add_argument("--label-suffix", default="_gtFine_labelIds.png")
    parser.add_argument("--list", help="Optional supplied split list, relative to the image directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    image_dir = root / args.images
    if args.list:
        names = [line.strip().replace("\\", "/") for line in Path(args.list).read_text().splitlines() if line.strip()]
        files = [image_dir / name for name in names]
    else:
        files = sorted(image_dir.rglob("*" + args.image_suffix))
    if not files:
        raise ValueError(f"No images found in {image_dir}")
    rows = []
    for image in files:
        if not image.is_file() or not image.name.endswith(args.image_suffix):
            raise FileNotFoundError(image)
        relative = image.relative_to(image_dir).as_posix()
        label = ""
        if args.labels:
            label_name = relative[:-len(args.image_suffix)] + args.label_suffix
            label_path = root / args.labels / label_name
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            label = label_path.relative_to(root).as_posix()
        rows.append({"image": image.relative_to(root).as_posix(), "label": label})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} entries to {output}")

def errors(argv):
    parser = argparse.ArgumentParser(description="Build supervised verification targets from source predictions")
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True, help="Source images and prepared source ground truth")
    parser.add_argument("--predictions", required=True, help="Prediction directory, relative to root")
    parser.add_argument("--errors", required=True, help="Error target directory, relative to root")
    parser.add_argument("--output", required=True, help="Verification manifest CSV")
    parser.add_argument("--color-predictions", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root)
    rows = read_manifest(args.manifest, require_labels=True)
    output_rows = []
    for row in rows:
        name = cache_name(row["image"])
        prediction_path = Path(args.predictions) / name
        with Image.open(root / prediction_path) as image:
            prediction = color_to_ids(np.array(image.convert("RGB"))) if args.color_predictions else np.array(image)
        validate_ids(prediction)
        with Image.open(root / row["label"]) as image:
            truth = np.array(image.resize((prediction.shape[1], prediction.shape[0]), Image.Resampling.NEAREST))
        validate_ids(truth)
        prediction[truth == 255] = 255
        error = ((prediction != truth) & (truth != 255)).astype(np.uint8) * 255
        error_path = Path(args.errors) / name
        (root / error_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(error).save(root / error_path)

        indexed_path = Path(args.errors) / "predictions" / name
        (root / indexed_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(prediction).save(root / indexed_path)
        output_rows.append({"image": row["image"], "prediction": indexed_path.as_posix(), "error": error_path.as_posix()})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "prediction", "error"])
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(rows)} verification examples to {output}")

def split(argv):
    parser = argparse.ArgumentParser(description="Save a reproducible train/validation manifest split")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not 0 < args.val_fraction < 1:
        raise ValueError("val-fraction must be between zero and one")
    with open(args.input, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    if len(rows) < 2:
        raise ValueError("At least two examples are needed for a split")
    random.Random(args.seed).shuffle(rows)
    count = max(1, min(len(rows) - 1, round(len(rows) * args.val_fraction)))
    for destination, selected in [(args.val, rows[:count]), (args.train, rows[count:])]:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
        print(f"{destination}: {len(selected)} examples")

def main():
    parser = argparse.ArgumentParser(description="Prepare ForeCASS labels and dataset lists")
    parser.add_argument("command", choices=["labels", "manifest", "errors", "split"])
    args = parser.parse_args(sys.argv[1:2])
    commands = {"labels": labels, "manifest": manifest, "errors": errors, "split": split}
    commands[args.command](sys.argv[2:])


if __name__ == "__main__":
    main()
