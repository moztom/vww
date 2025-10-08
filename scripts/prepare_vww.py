"""
End-to-end Visual Wake Words (VWW) data prep.

What it does (in one run):
1) Downloads MS COCO (2014/2017) via the helper scripts packaged with `pyvww`.
2) Creates the official COCO maxitrain/minival split.
3) Generates VWW (binary person/not-person) annotations.
4) Exports images resized to 96x96 into:
   <export_dir>/{train,val}/{0,1}/*.jpg
   where 1 = person present, 0 = no person.

Usage:
  pip install pyvww pycocotools pillow tqdm
  python prep_vww_96.py \
    --coco_dir ../data/coco \
    --vww_dir ../data/vww \
    --export_dir ../data/vww96 \
    --year 2017
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image
from tqdm import tqdm
import pyvww


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ensure_coco(coco_dir: Path, year: str) -> None:
    """
    Ensure MS COCO images + annotations exist locally. If not, download them
    using the `download_mscoco.sh` script that ships with pyvww.

    Args:
        pyvww_scripts: Path to the pyvww/scripts directory.
        coco_dir: Destination directory for the COCO dataset.
        year: "2014" or "2017" split to download.
    """
    ann_dir = coco_dir / "annotations"

    # Quick presence check: val images + annotation folder are a good proxy.
    #has_images = (coco_dir / f"val{year}").exists() or (coco_dir / f"train{year}").exists()
    #not has_images or 
    if not ann_dir.exists():
        coco_dir.mkdir(parents=True, exist_ok=True)
        print("Starting COCO download")
        run(["bash", "pyvww/download_mscoco.sh", str(coco_dir), str(year)])
    else:
        print("COCO already downloaded, skipping")


def make_splits(coco_dir: Path, year: str) -> Tuple[Path, Path]:
    """
    Create the COCO "maxitrain" (train) and "minival" (val) annotation JSONs
    using the pyvww helper script. If they already exist, reuse them.

    Args:
        pyvww_scripts: Path to the pyvww/scripts directory.
        coco_dir: Root of the COCO dataset.
        year: "2014" or "2017".

    Returns:
        (maxitrain_json_path, minival_json_path)
    """
    ann_dir = coco_dir / "annotations"
    maxitrain = ann_dir / "instances_maxitrain.json"
    minival = ann_dir / "instances_minival.json"

    if maxitrain.exists() and minival.exists():
        print("Split JSONs already exist, skipping")
        return maxitrain, minival

    run([
        sys.executable, "pyvww/create_coco_train_minival_split.py",
        f"--train_annotations_file={ann_dir / f'instances_train{year}.json'}",
        f"--val_annotations_file={ann_dir / f'instances_val{year}.json'}",
        f"--output_dir={ann_dir}",
    ])
    return maxitrain, minival


def make_vww_ann(
    maxitrain: Path,
    minival: Path,
    out_dir: Path,
    threshold: float,
    foreground: str,
) -> Tuple[Path, Path]:
    """
    Generate Visual Wake Words annotations (binary labels) from the COCO split.

    The helper script assigns label 1 to images where the given `foreground`
    class (default "person") occupies at least `threshold` fraction of the image.

    Args:
        pyvww_scripts: Path to pyvww/scripts.
        maxitrain: Path to instances_maxitrain.json (COCO-like).
        minival: Path to instances_minival.json (COCO-like).
        out_dir: Directory where VWW annotations will be written.
        threshold: Area ratio threshold for the foreground class.
        foreground: Foreground class name, usually "person".

    Returns:
        (vww_train_json_path, vww_val_json_path)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    vww_train = out_dir / "instances_train.json"
    vww_val = out_dir / "instances_val.json"

    if vww_train.exists() and vww_val.exists():
        print("Annotations already exist, skipping")
        return vww_train, vww_val

    run([
        sys.executable, "pyvww/create_visualwakewords_annotations.py",
        f"--train_annotations_file={maxitrain}",
        f"--val_annotations_file={minival}",
        f"--output_dir={out_dir}",
        f"--threshold={threshold}",
        f"--foreground_class={foreground}",
    ])
    return vww_train, vww_val


def load_coco_like(json_path: Path) -> Tuple[Dict[int, dict], Dict[int, int]]:
    """
    Load a COCO-like annotation file and build quick lookup maps.

    Args:
        json_path: Path to the annotation JSON.

    Returns:
        id_to_image: {image_id -> image_record}
        image_to_label: {image_id -> 0 or 1}
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Map image id -> image dict (contains file_name, width, height, etc.)
    id_to_image = {img["id"]: img for img in data["images"]}

    # For VWW, annotations are 1-per-image; category_id is the binary label.
    image_to_label: Dict[int, int] = {}
    for ann in data.get("annotations", []):
        image_to_label[ann["image_id"]] = int(ann["category_id"])

    return id_to_image, image_to_label


def export_split(
    json_path: Path,
    coco_dir: Path,
    year: str,
    out_root: Path,
    split_name: str,
    size: Tuple[int, int] = (96, 96),
) -> None:
    """
    Export a resized image set (e.g., 96x96) for one split (train or val),
    saving to a class-balanced folder layout: <out_root>/<split_name>/{0,1}/*.jpg

    Args:
        json_path: Path to VWW COCO-like annotations for this split.
        coco_dir: Root of the COCO dataset.
        year: "2014" or "2017".
        out_root: Root directory where resized images will be written.
        split_name: "train" or "val".
        size: (width, height) to resize to; default (96,96) for VWW.
    """
    id_to_image, image_to_label = load_coco_like(json_path)

    # Create output dirs (one per class).
    split_out_0 = out_root / split_name / "0"  # 0 = no person
    split_out_1 = out_root / split_name / "1"  # 1 = person
    split_out_0.mkdir(parents=True, exist_ok=True)
    split_out_1.mkdir(parents=True, exist_ok=True)

    def resolve_path(file_name: str) -> Path | None:
        """
        Try to resolve an image path. COCO stores images under train{year}/ and val{year}/.
        Some file_name entries may already include subfolders.
        """
        p = coco_dir / f"train{year}" / file_name
        if p.exists():
            return p
        p = coco_dir / f"val{year}" / file_name
        if p.exists():
            return p
        p = coco_dir / file_name  # fallback if file_name includes subdir
        return p if p.exists() else None

    missing = 0

    # Iterate images referenced by the annotation JSON.
    for img_id, img_meta in tqdm(id_to_image.items(), desc=f"Export {split_name}"):
        label = image_to_label.get(img_id)
        if label is None:
            # Should not happen for VWW, but be robust to malformed labels.
            continue

        # Figure out where the original COCO image lives.
        src = resolve_path(img_meta["file_name"])
        if src is None:
            missing += 1
            continue

        # Destination path mirrors the original stem, but we fix extension to .jpg
        dst_dir = split_out_1 if label == 1 else split_out_0
        dst = dst_dir / (Path(img_meta["file_name"]).stem + ".jpg")

        if dst.exists():
            # If already exported, skip it.
            continue

        try:
            # Open, convert to RGB (handles PNGs/CMYK), resize with high-quality filter, save JPEG.
            Image.open(src).convert("RGB")\
                .resize(size, Image.LANCZOS)\
                .save(dst, format="JPEG", quality=95)
        except Exception:
            # Corrupt/unreadable files get skipped silently; training can tolerate a few misses.
            continue

    if missing:
        print(f"[warn] {missing} images referenced but not found; skipped.", file=sys.stderr)


def main() -> None:
    """
    Parse CLI args, run the full pipeline, and print where the ready-to-train
    folders ended up.
    """
    ap = argparse.ArgumentParser(description="End-to-end VWW prep with 96x96 export")
    ap.add_argument("--coco_dir", type=Path, required=True, help="Where COCO will live")
    ap.add_argument("--vww_dir", type=Path, required=True, help="Where VWW annotations will be written")
    ap.add_argument("--export_dir", type=Path, required=True, help="Where 96x96 images will be exported")
    ap.add_argument("--year", choices=["2014", "2017"], default="2017", help="COCO split to use")
    ap.add_argument("--threshold", type=float, default=0.005, help="Area ratio for foreground presence (e.g., 0.5%)")
    ap.add_argument("--foreground", default="person", help="Foreground class to detect (default: person)")
    args = ap.parse_args()

    # 1) Fetch COCO if needed
    ensure_coco(args.coco_dir, args.year)

    # 2) Make maxitrain/minival split
    maxitrain, minival = make_splits(args.coco_dir, args.year)

    # 3) Generate VWW annotations (binary labels)
    vww_ann_dir = args.vww_dir / "annotations"
    vww_train, vww_val = make_vww_ann(
        maxitrain, minival, vww_ann_dir, args.threshold, args.foreground
    )

    # 4) Export resized images for both splits
    export_split(vww_train, args.coco_dir, args.year, args.export_dir, "train", size=(96, 96))
    export_split(vww_val, args.coco_dir, args.year, args.export_dir, "val", size=(96, 96))

    print("\nDone. Folders ready for training:")
    print(f"  {args.export_dir}/train/0  (no-person)")
    print(f"  {args.export_dir}/train/1  (person)")
    print(f"  {args.export_dir}/val/0")
    print(f"  {args.export_dir}/val/1")


if __name__ == "__main__":
    main()
