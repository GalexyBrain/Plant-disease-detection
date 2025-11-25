#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_cnn_per_class.py

Per-class feature-map inspection for your DeeperCNN:
  • Samples N images per class
  • Saves top-activation feature maps for block1/2/3 for each image
  • Builds per-class average feature maps (mean over images) and saves those too

Usage:
  python inspect_cnn_per_class.py --model_path ... --data_dir dataset --subset val --n_per_class 3 --topk 16
"""

import os, math, csv, random, argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- Args ----------------------

def get_args():
    p = argparse.ArgumentParser("Per-class feature map inspector")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="dataset")
    p.add_argument("--subset", type=str, default="val", choices=["train","val","test"])
    p.add_argument("--img_size", type=int, default=112)
    p.add_argument("--n_per_class", type=int, default=2, help="images per class")
    p.add_argument("--topk", type=int, default=16, help="top channels to visualize")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()

# ---------------------- Model ----------------------

class DeeperCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 5, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.3),

            nn.Conv2d(128, 128, 7, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.35),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*14*14, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------------- Utils ----------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def dynamic_outdir(tag="PerClass"):
    t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = f"{tag}_{t}"
    Path(out).mkdir(parents=True, exist_ok=True)
    return out

def make_grid(rows, cols, cell=3.0):
    fig, axes = plt.subplots(rows, cols, figsize=(cols*cell, rows*cell))
    if rows == 1 and cols == 1: axes = np.array([[axes]])
    elif rows == 1:             axes = np.array([axes])
    elif cols == 1:             axes = np.array([[ax] for ax in axes])
    return fig, axes

def plot_topk_featuremaps(fmap_chw: torch.Tensor, topk: int, save_path: str, title: str):
    """
    fmap_chw: [C,H,W] tensor (CPU)
    """
    C, H, W = fmap_chw.shape
    k = min(topk, C)
    mean_per_ch = fmap_chw.mean(dim=(1,2))
    top_idx = torch.topk(mean_per_ch, k=k).indices.cpu().numpy()

    cols = 4
    rows = math.ceil(k/cols)
    fig, axes = make_grid(rows, cols, cell=3.2)

    for i, ch in enumerate(top_idx):
        r, c = divmod(i, cols)
        arr = fmap_chw[int(ch)].numpy()
        vmin, vmax = np.percentile(arr, 1), np.percentile(arr, 99)
        arr = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0, 1)
        axes[r, c].imshow(arr, cmap="viridis")
        axes[r, c].set_title(f"ch {int(ch)}", fontsize=8)
        axes[r, c].axis("off")

    # hide unused cells
    for j in range(k, rows*cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# ---------------------- Main ----------------------

def main():
    args = get_args()
    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = args.output_dir or dynamic_outdir("PerClass")

    # Dataset
    subset_dir = Path(args.data_dir) / args.subset
    if not subset_dir.exists():
        raise FileNotFoundError(f"Subset not found: {subset_dir}")

    base_ds = datasets.ImageFolder(str(subset_dir), transform=None)
    class_names = base_ds.classes
    num_classes = len(class_names)
    print(f"Found classes: {class_names}")

    # Transforms identical to training
    full_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Model
    model = DeeperCNN(num_classes=num_classes)
    state = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(device)

    # Feature-map hooks (after BN in each block)
    hook_layers = {
        "block1": model.features[5],   # BN after 2nd conv
        "block2": model.features[13],  # BN after 2nd conv
        "block3": model.features[18],  # BN after last conv
    }
    fmap_store = {}
    hooks = []
    def save_activation(name):
        def fn(_, __, output):
            fmap_store[name] = output.detach().cpu()  # [B,C,H,W]
        return fn
    for name, layer in hook_layers.items():
        hooks.append(layer.register_forward_hook(save_activation(name)))

    softmax = nn.Softmax(dim=1)

    # Index images per class
    images_by_class = {i: [] for i in range(num_classes)}
    for idx, (path, lbl) in enumerate(base_ds.samples):
        images_by_class[lbl].append(path)

    # Iterate classes
    for cls_idx, cls_name in enumerate(class_names):
        cls_out = Path(out_root) / f"{cls_idx:02d}_{cls_name}"
        cls_out.mkdir(parents=True, exist_ok=True)

        paths = images_by_class[cls_idx]
        if len(paths) == 0:
            print(f"[WARN] No images for class {cls_name}")
            continue

        random.shuffle(paths)
        chosen = paths[:min(args.n_per_class, len(paths))]

        # Track per-class sums to build average maps
        class_sums = {"block1": None, "block2": None, "block3": None}
        used = []

        # Per-image dumps
        for pth in chosen:
            img = Image.open(pth).convert("RGB")
            x = full_transform(img).unsqueeze(0).to(device)

            fmap_store.clear()
            with torch.no_grad():
                logits = model(x)
                _ = softmax(logits)  # just to be consistent; not needed for maps

            # fmap_store[...] = [1,C,H,W]
            for b in ["block1", "block2", "block3"]:
                fmap = fmap_store[b].squeeze(0)  # [C,H,W] CPU

                # Add to sums
                if class_sums[b] is None:
                    class_sums[b] = fmap.clone()
                else:
                    # accumulate (same shapes by construction)
                    class_sums[b] += fmap

                # Save per-image grid
                title = f"Top-activation feature maps — {b} | {Path(pth).name} ({cls_name})"
                save_path = cls_out / f"featuremaps_{b}_{Path(pth).stem}.png"
                plot_topk_featuremaps(fmap, args.topk, str(save_path), title)

            used.append(pth)

        # Per-class average maps
        n_used = max(1, len(used))
        for b in ["block1", "block2", "block3"]:
            avg_map = class_sums[b] / n_used  # [C,H,W]
            title = f"Per-class AVERAGE feature maps — {b} | class={cls_name} (N={n_used})"
            save_path = cls_out / f"featuremaps_{b}_CLASS_AVG.png"
            plot_topk_featuremaps(avg_map, args.topk, str(save_path), title)

        # Save which images were used
        with open(cls_out / "picked_images.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["class", "path"])
            for p in used: w.writerow([cls_name, p])

        print(f"[{cls_name}] wrote {len(used)} per-image grids + class averages → {cls_out}")

    # Cleanup hooks
    for h in hooks: h.remove()
    print(f"\n✅ Done. Outputs in: {out_root}")

if __name__ == "__main__":
    main()
