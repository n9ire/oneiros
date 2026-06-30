"""
Computer Vision dataset utilities for Oneiros.

Provides:
  load_image_folder(path)              → metadata + thumbnails
  make_image_loaders(session_id, ...)  → (train_loader, val_loader)
  build_augmentation_transform(steps)  → transforms.Compose
  preview_augmentation(session_id, ...) → PNG bytes
  AUG_STEP_META                         → palette definitions for the frontend
"""

from __future__ import annotations

import base64
import io
import os
import uuid
import zipfile
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split

# torchvision is required for CV support
def _tv():
    try:
        import torchvision
        return torchvision
    except ImportError:
        raise RuntimeError("torchvision is not installed. Run: pip install torchvision")

def _tv_transforms():
    return _tv().transforms

def _tv_datasets():
    return _tv().datasets


# ── Session store ─────────────────────────────────────────────────────────────

_cv_sessions: dict[str, dict[str, Any]] = {}

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


# ── Load ──────────────────────────────────────────────────────────────────────

def load_image_folder(path: str | Path) -> dict:
    """
    Scan an ImageFolder-style directory and return rich metadata.

    Expected layout:
        root/
          class_a/img1.jpg
          class_b/img1.png
          ...

    Returns
    -------
    {
      sessionId, name,
      classNames, classCounts, totalImages,
      inputShape: [C, H, W],   # inferred from first found image
      thumbnails: { className: [base64_jpeg, ...] },   # 4 per class
      warnings: list[str],
    }
    """
    tv_transforms = _tv_transforms()
    path = Path(path)

    # Discover class directories
    class_dirs = sorted([d for d in path.iterdir() if d.is_dir()])
    if len(class_dirs) == 0:
        raise ValueError(
            "No class subdirectories found. "
            "Your zip must have one folder per class: class_a/img1.jpg, class_b/img2.jpg …"
        )
    if len(class_dirs) == 1:
        raise ValueError(
            f"Only 1 class found ('{class_dirs[0].name}'). "
            "At least 2 classes are required for classification. "
            "Add another class folder to your zip."
        )

    class_names: list[str] = []
    class_counts: dict[str, int] = {}
    thumbnails: dict[str, list[str]] = {}
    warnings: list[str] = []
    all_sizes: list[tuple[int, int]] = []
    inferred_channels = 3

    for cls_dir in class_dirs:
        name = cls_dir.name
        img_files = [
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTS
        ]
        if not img_files:
            warnings.append(
                f"Class '{name}' contains no supported images (.jpg/.png/.bmp/.tiff/.webp) — "
                "it will be ignored unless you add images."
            )
            continue

        class_names.append(name)
        class_counts[name] = len(img_files)

        # Collect thumbnails (first 4 images)
        thumbs: list[str] = []
        for img_file in img_files[:4]:
            try:
                from PIL import Image as PILImage
                img = PILImage.open(img_file)
                all_sizes.append(img.size)  # (W, H)
                # Infer channels from first image
                if img.mode == "L":
                    inferred_channels = 1
                elif img.mode == "RGBA":
                    inferred_channels = 4
                else:
                    inferred_channels = 3
                # Thumbnail
                thumb = img.convert("RGB").resize((64, 64))
                buf = io.BytesIO()
                thumb.save(buf, format="JPEG", quality=75)
                thumbs.append(base64.b64encode(buf.getvalue()).decode())
            except Exception:
                pass
        thumbnails[name] = thumbs

    if not class_names:
        raise ValueError(
            "All class folders are empty — no images found. "
            "Add .jpg/.png images inside each class folder."
        )

    # Infer dominant image size
    inferred_h, inferred_w = 224, 224  # safe default
    if all_sizes:
        most_common = max(set(all_sizes), key=all_sizes.count)
        inferred_w, inferred_h = most_common
        min_w = min(s[0] for s in all_sizes)
        min_h = min(s[1] for s in all_sizes)
        max_w = max(s[0] for s in all_sizes)
        max_h = max(s[1] for s in all_sizes)
        if min_w != max_w or min_h != max_h:
            warnings.append(
                f"Image sizes vary: {min_w}–{max_w} × {min_h}–{max_h} px. "
                "Add a Resize augmentation node to standardise dimensions before training."
            )

    total = sum(class_counts.values())
    session_id = str(uuid.uuid4())
    _cv_sessions[session_id] = {
        "path": str(path),
        "class_names": class_names,
        "class_counts": class_counts,
        "input_shape": (inferred_channels, inferred_h, inferred_w),
        "name": path.name,
    }

    return {
        "sessionId":   session_id,
        "name":        path.name,
        "classNames":  class_names,
        "classCounts": class_counts,
        "totalImages": total,
        "inputShape":  [inferred_channels, inferred_h, inferred_w],
        "thumbnails":  thumbnails,
        "warnings":    warnings,
    }


# ── Augmentation ──────────────────────────────────────────────────────────────

def build_augmentation_transform(steps: list[dict]) -> Any:
    """
    Build a torchvision.transforms.Compose from an ordered list of step dicts.
    Each step: {"type": str, **params}.
    """
    T = _tv_transforms()
    transform_list = []

    for step in steps:
        stype = step.get("type", "")

        if stype == "resize":
            w = int(step.get("width", 224))
            h = int(step.get("height", 224))
            transform_list.append(T.Resize((h, w)))

        elif stype == "randomCrop":
            size = int(step.get("size", 224))
            pad = int(step.get("padding", 0))
            transform_list.append(T.RandomCrop(size, padding=pad if pad > 0 else None))

        elif stype == "centerCrop":
            size = int(step.get("size", 224))
            transform_list.append(T.CenterCrop(size))

        elif stype == "randomHFlip":
            p = float(step.get("probability", 0.5))
            transform_list.append(T.RandomHorizontalFlip(p=p))

        elif stype == "randomVFlip":
            p = float(step.get("probability", 0.5))
            transform_list.append(T.RandomVerticalFlip(p=p))

        elif stype == "colorJitter":
            b = float(step.get("brightness", 0.2))
            c = float(step.get("contrast", 0.2))
            s = float(step.get("saturation", 0.2))
            h = float(step.get("hue", 0.0))
            transform_list.append(T.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h))

        elif stype == "randomRotation":
            deg = float(step.get("degrees", 15))
            transform_list.append(T.RandomRotation(degrees=deg))

        elif stype == "gaussianBlur":
            k = int(step.get("kernelSize", 3))
            # kernelSize must be odd
            if k % 2 == 0:
                k += 1
            sigma = step.get("sigma", None)
            sig = (float(sigma),) * 2 if sigma else (0.1, 2.0)
            transform_list.append(T.GaussianBlur(kernel_size=k, sigma=sig))

        elif stype == "normalize":
            mean_r = float(step.get("meanR", 0.485))
            mean_g = float(step.get("meanG", 0.456))
            mean_b = float(step.get("meanB", 0.406))
            std_r  = float(step.get("stdR",  0.229))
            std_g  = float(step.get("stdG",  0.224))
            std_b  = float(step.get("stdB",  0.225))
            transform_list.append(T.Normalize(mean=[mean_r, mean_g, mean_b], std=[std_r, std_g, std_b]))

    # Always append ToTensor if not already added (must come before Normalize)
    # Insert ToTensor before the first Normalize if present
    has_to_tensor = any(isinstance(t, T.ToTensor) for t in transform_list)  # type: ignore[attr-defined]
    if not has_to_tensor:
        # Insert before Normalize
        norm_idx = next((i for i, t in enumerate(transform_list) if isinstance(t, T.Normalize)), len(transform_list))  # type: ignore[attr-defined]
        transform_list.insert(norm_idx, T.ToTensor())

    return T.Compose(transform_list)


# ── Default (no augmentation) transform ──────────────────────────────────────

def _default_transform(input_shape: tuple[int, int, int]) -> Any:
    """Standard inference transform: resize to input_shape then ToTensor."""
    T = _tv_transforms()
    _, h, w = input_shape
    return T.Compose([T.Resize((h, w)), T.ToTensor()])


# ── DataLoader factory ────────────────────────────────────────────────────────

def make_image_loaders(
    session_id: str,
    augment_steps: list[dict],
    batch_size: int = 32,
    val_split: float = 0.2,
) -> tuple[DataLoader, DataLoader]:
    if session_id not in _cv_sessions:
        raise ValueError(
            f"CV session '{session_id}' not found. "
            "Upload your image zip again — sessions reset when the backend restarts."
        )

    sess = _cv_sessions[session_id]
    input_shape: tuple[int, int, int] = sess["input_shape"]

    if augment_steps:
        train_transform = build_augmentation_transform(augment_steps)
    else:
        train_transform = _default_transform(input_shape)

    val_transform = _default_transform(input_shape)

    ImageFolder = _tv_datasets().ImageFolder

    try:
        full_dataset = ImageFolder(sess["path"], transform=train_transform)
    except FileNotFoundError:
        raise ValueError(
            "Image dataset path not found. "
            "The session may have expired — re-upload your zip file."
        )

    if len(full_dataset) == 0:
        raise ValueError(
            "No images found in the dataset directory. "
            "Ensure your zip contains class subfolders with .jpg/.png images."
        )

    n_val = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val

    if n_train < batch_size:
        raise ValueError(
            f"Not enough training images ({n_train}) for batch size {batch_size}. "
            "Reduce batch size or add more images."
        )

    train_set, val_set_raw = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Val set uses inference-only transform (no augmentation)
    val_dataset = ImageFolder(sess["path"], transform=val_transform)
    val_indices = val_set_raw.indices
    val_set = torch.utils.data.Subset(val_dataset, val_indices)

    num_workers = 0
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


# ── Augmentation preview ──────────────────────────────────────────────────────

def preview_augmentation(
    session_id: str,
    steps: list[dict],
    image_index: int = 0,
) -> bytes:
    """
    Apply the augmentation pipeline to one source image and return a side-by-side
    before/after PNG as bytes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    if session_id not in _cv_sessions:
        raise ValueError(f"Session '{session_id}' not found.")

    sess = _cv_sessions[session_id]
    root = Path(sess["path"])

    # Find the requested image
    imgs: list[Path] = []
    for cls in sess["class_names"]:
        cls_dir = root / cls
        imgs.extend(sorted(
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTS
        ))

    if not imgs:
        raise ValueError("No images found in session.")
    img_path = imgs[image_index % len(imgs)]

    T = _tv_transforms()
    original = PILImage.open(img_path).convert("RGB")

    # Build transform (without ToTensor so we get PIL back for display)
    pil_steps = [s for s in steps if s.get("type") != "normalize"]
    import torchvision.transforms.functional as TF

    # Apply PIL-compatible transforms
    aug = original.copy()
    for step in pil_steps:
        stype = step.get("type", "")
        if stype == "resize":
            aug = aug.resize((int(step.get("width", 224)), int(step.get("height", 224))))
        elif stype == "randomHFlip":
            import random
            if random.random() < float(step.get("probability", 0.5)):
                aug = TF.hflip(aug)
        elif stype == "randomVFlip":
            import random
            if random.random() < float(step.get("probability", 0.5)):
                aug = TF.vflip(aug)
        elif stype == "centerCrop":
            aug = TF.center_crop(aug, int(step.get("size", 224)))
        elif stype == "randomRotation":
            import random
            deg = float(step.get("degrees", 15))
            aug = TF.rotate(aug, random.uniform(-deg, deg))
        elif stype == "colorJitter":
            aug = T.ColorJitter(
                brightness=float(step.get("brightness", 0.2)),
                contrast=float(step.get("contrast", 0.2)),
                saturation=float(step.get("saturation", 0.2)),
                hue=float(step.get("hue", 0.0)),
            )(aug)
        elif stype == "gaussianBlur":
            k = int(step.get("kernelSize", 3))
            if k % 2 == 0:
                k += 1
            aug = T.GaussianBlur(kernel_size=k)(aug)

    DARK = "#09090b"
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor=DARK)
    for ax, im, title in [(axes[0], original, "Original"), (axes[1], aug, "Augmented")]:
        ax.imshow(im)
        ax.set_title(title, color="#d4d4d8", fontsize=11, pad=6)
        ax.axis("off")
        ax.set_facecolor(DARK)
        for spine in ax.spines.values():
            spine.set_edgecolor("#27272a")
    fig.suptitle(img_path.parent.name, color="#71717a", fontsize=9)
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=DARK, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Extract zip ───────────────────────────────────────────────────────────────

def extract_zip_to_temp(zip_bytes: bytes, name: str) -> Path:
    """Extract a zip to a persistent temp directory and return the root path."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="oneiros_cv_"))
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # Security: reject absolute paths and path traversal
        for member in zf.namelist():
            if member.startswith("/") or ".." in member:
                raise ValueError(f"Unsafe zip entry: {member!r}")
        zf.extractall(tmp_dir)

    # If there's a single top-level directory, descend into it
    entries = list(tmp_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return tmp_dir


import tempfile  # noqa: E402  (must come after Path import)


# ── Palette meta ──────────────────────────────────────────────────────────────

AUG_STEP_META = [
    {
        "type": "resize",
        "label": "Resize",
        "params": [
            {"key": "width",  "label": "Width",  "type": "number", "default": 224},
            {"key": "height", "label": "Height", "type": "number", "default": 224},
        ],
    },
    {
        "type": "randomCrop",
        "label": "Random Crop",
        "params": [
            {"key": "size",    "label": "Size",    "type": "number", "default": 224},
            {"key": "padding", "label": "Padding", "type": "number", "default": 0},
        ],
    },
    {
        "type": "centerCrop",
        "label": "Center Crop",
        "params": [
            {"key": "size", "label": "Size", "type": "number", "default": 224},
        ],
    },
    {
        "type": "randomHFlip",
        "label": "Random H-Flip",
        "params": [
            {"key": "probability", "label": "Probability", "type": "number", "default": 0.5},
        ],
    },
    {
        "type": "randomVFlip",
        "label": "Random V-Flip",
        "params": [
            {"key": "probability", "label": "Probability", "type": "number", "default": 0.5},
        ],
    },
    {
        "type": "colorJitter",
        "label": "Color Jitter",
        "params": [
            {"key": "brightness", "label": "Brightness", "type": "number", "default": 0.2},
            {"key": "contrast",   "label": "Contrast",   "type": "number", "default": 0.2},
            {"key": "saturation", "label": "Saturation", "type": "number", "default": 0.2},
            {"key": "hue",        "label": "Hue",        "type": "number", "default": 0.0},
        ],
    },
    {
        "type": "randomRotation",
        "label": "Random Rotation",
        "params": [
            {"key": "degrees", "label": "Degrees", "type": "number", "default": 15},
        ],
    },
    {
        "type": "gaussianBlur",
        "label": "Gaussian Blur",
        "params": [
            {"key": "kernelSize", "label": "Kernel Size", "type": "number", "default": 3},
            {"key": "sigma",      "label": "Sigma",       "type": "number", "default": 1.0},
        ],
    },
    {
        "type": "normalize",
        "label": "Normalize",
        "params": [
            {"key": "meanR", "label": "Mean R", "type": "number", "default": 0.485},
            {"key": "meanG", "label": "Mean G", "type": "number", "default": 0.456},
            {"key": "meanB", "label": "Mean B", "type": "number", "default": 0.406},
            {"key": "stdR",  "label": "Std R",  "type": "number", "default": 0.229},
            {"key": "stdG",  "label": "Std G",  "type": "number", "default": 0.224},
            {"key": "stdB",  "label": "Std B",  "type": "number", "default": 0.225},
        ],
    },
]
