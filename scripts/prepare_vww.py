"""
End-to-end Visual Wake Words (VWW) data prep.

What it does (in one run):
1) Downloads MS COCO (2014/2017) via helper scripts vendored under scripts/pyvww.
2) Creates the official COCO maxitrain/minival split.
3) Generates VWW (binary person/not-person) annotations.
4) Exports images resized to {size}x{size} into:
   data/vww{size}/{train,val}/{0,1}/*.jpg
   where 1 = person present, 0 = no person.

Usage:
  pip install pyvww pycocotools pillow tqdm
  python scripts/prepare_vww.py --year 2017 --size 96
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image, ImageOps
from tqdm import tqdm

# Local copy of pyvww scripts
PYVWW_SCRIPTS_DIR = Path(__file__).resolve().parent / "pyvww"

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
DEFAULT_COCO_DIR = DATA_ROOT / "coco"
DEFAULT_VWW_DIR = DATA_ROOT / "vww"

# IMAGENET1K_V2
IMAGENET_MEAN = [0.485, 0.456, 0.406]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ensure_coco(coco_dir: Path, year: str) -> None:
    """
    Ensure MS COCO images + annotations exist locally. If not, download them
    using the helper shell script checked into scripts/pyvww.

    Args:
        coco_dir: Destination directory for the COCO dataset.
        year: "2014" or "2017" split to download.
    """
    ann_dir = coco_dir / "annotations"

    # Quick presence check: val images + annotation folder are a good proxy.
    #has_images = (coco_dir / f"val{year}").exists() or (coco_dir / f"train{year}").exists()
    #not has_images or 
    if not ann_dir.exists():
        coco_dir.mkdir(parents=True, exist_ok=True)
        run(["bash", str(PYVWW_SCRIPTS_DIR / "download_mscoco.sh"), str(coco_dir), str(year)])
    else:
        print("COCO already downloaded, skipping")


def make_splits(coco_dir: Path, year: str) -> Tuple[Path, Path]:
    """
    Create the COCO "maxitrain" (train) and "minival" (val) annotation JSONs
    using the helper script stored alongside this script.

    Args:
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
        sys.executable, str(PYVWW_SCRIPTS_DIR / "create_coco_train_minival_split.py"),
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
) -> Tuple[Path, Path]:
    """
    Generate Visual Wake Words annotations (binary labels) from the COCO split.

    The helper script assigns label 1 to images where the `foreground`
    class ("person") occupies at least `threshold` fraction of the image.

    Args:
        maxitrain: Path to instances_maxitrain.json (COCO-like).
        minival: Path to instances_minival.json (COCO-like).
        out_dir: Directory where VWW annotations will be written.
        threshold: Area ratio threshold for the foreground class.

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
        sys.executable, str(PYVWW_SCRIPTS_DIR / "create_visualwakewords_annotations.py"),
        f"--train_annotations_file={maxitrain}",
        f"--val_annotations_file={minival}",
        f"--output_dir={out_dir}",
        f"--threshold={threshold}",
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
    id_to_image = {img["id"]: img for img in data.get("images", [])}

    # Map image id -> label (0 or 1). Default to 0 (no person).
    image_to_label: Dict[int, int] = {img_id: 0 for img_id in id_to_image}
    for ann in data.get("annotations", []):
        if int(ann["category_id"]) == 1:
            image_to_label[ann["image_id"]] = 1

    return id_to_image, image_to_label


def export_split(
    json_path: Path,
    coco_dir: Path,
    year: str,
    out_root: Path,
    split_name: str,
    size: Tuple[int, int],
    mode: str,
    pad_rgb: Tuple[int, int, int] | None = None,
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
        size: (width, height) to resize to.
        mode: export policy:
            - "center-crop": AR-preserving resize (short side -> target), then center crop
            - "letterbox"  : AR-preserving resize (long  side -> target), then pad to square
            - "resize"     : legacy direct resize to (w,h)
        pad_rgb: RGB tuple for padding when mode="letterbox".
    """
    id_to_image, image_to_label = load_coco_like(json_path)

    # Create output dirs (one per class).
    split_out_0 = out_root / split_name / "0"  # 0 = no person
    split_out_1 = out_root / split_name / "1"  # 1 = person
    split_out_0.mkdir(parents=True, exist_ok=True)
    split_out_1.mkdir(parents=True, exist_ok=True)

    def _resolve_path(file_name: str) -> Path | None:
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

    def _center_crop_after_resize(img: Image.Image, target: int) -> Image.Image:
        # resize so short side == target, then center crop to target×target
        w, h = img.size
        scale = target / min(w, h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img = img.resize((nw, nh), resample=Image.LANCZOS)
        left = (nw - target) // 2
        top = (nh - target) // 2
        return img.crop((left, top, left + target, top + target))
 
    def _letterbox_to_square(img: Image.Image, target: int, rgb: Tuple[int, int, int]) -> Image.Image:
        # resize so long side == target, then pad to target×target
        w, h = img.size
        scale = target / max(w, h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img = img.resize((nw, nh), resample=Image.LANCZOS)
        pad_w, pad_h = target - nw, target - nh
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return ImageOps.expand(img, border=padding, fill=rgb)

    # Iterate images referenced by the annotation JSON.
    for img_id, img_meta in tqdm(id_to_image.items(), desc=f"Export {split_name}"):
        label = image_to_label.get(img_id)
        if label is None:
            # Should not happen for VWW, but be robust to malformed labels.
            continue

        # Figure out where the original COCO image lives.
        src = _resolve_path(img_meta["file_name"])
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
            img = Image.open(src).convert("RGB")
            if mode == "center-crop":
                out = _center_crop_after_resize(img, size[0])
            elif mode == "letterbox":
                out = _letterbox_to_square(img, size[0], pad_rgb)
            out.save(dst, format="JPEG", quality=95)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", choices=["2014", "2017"], default="2017", help="COCO split to use")
    ap.add_argument("--threshold", type=float, default=0.005, help="Area ratio for foreground presence (e.g., 0.5%)")
    ap.add_argument("--size", choices=[96, 224], type=int, required=True)
    args = ap.parse_args()

    coco_dir = DEFAULT_COCO_DIR
    vww_dir = DEFAULT_VWW_DIR
    export_dir = DATA_ROOT / f"vww{str(args.size)}"
    print(export_dir)

    # 1) Fetch COCO if needed
    ensure_coco(coco_dir, args.year)

    # 2) Make maxitrain/minival split
    maxitrain, minival = make_splits(coco_dir, args.year)

    # 3) Generate VWW annotations (binary labels)
    vww_ann_dir = vww_dir / "annotations"
    vww_train, vww_val = make_vww_ann(
        maxitrain, minival, vww_ann_dir, args.threshold
    )

    # 4) Export resized images for both splits
    
    if args.size == 96:
        mode = "letterbox"
        pad_rgb = tuple(int(round(m * 255)) for m in IMAGENET_MEAN)
        print(f"[info] using ImageNet mean for padding: {pad_rgb}")
    else:
        mode = "center-crop"
        pad_rgb = None

    export_split(vww_train, coco_dir, args.year, export_dir, "train", size=(args.size, args.size),
                 mode=mode, pad_rgb=pad_rgb)
    export_split(vww_val, coco_dir, args.year, export_dir, "val", size=(args.size, args.size),
                 mode=mode, pad_rgb=pad_rgb)

    print("\nDone. Folders ready for training:")
    print(f"  {export_dir}/train/0  (no-person)")
    print(f"  {export_dir}/train/1  (person)")
    print(f"  {export_dir}/val/0")
    print(f"  {export_dir}/val/1")


if __name__ == "__main__":
    main()
