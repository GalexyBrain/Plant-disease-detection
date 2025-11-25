import os
import shutil
from glob import glob
from tqdm import tqdm
import concurrent.futures

import torch
import numpy as np
import cv2
from torchvision.transforms import functional as TF
import kornia.augmentation as K
import albumentations as A

# ----------------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------------
NUM_AUGMENTS = 3
BATCH_SIZE   = 256
ORIG_ROOT    = "dataset"
AUG_ROOT     = "augmented_dataset"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE  = (224, 224)  # uniform size for batching
MAX_WORKERS  = os.cpu_count() or 8  # Number of threads for CPU steps

print(f"Started with the following : \n DEVICE = {DEVICE} \n MAX WORKERS = {MAX_WORKERS}")
print('-'*15)
# Infer classes from training folder
CLASSES = [os.path.basename(d) for d in sorted(glob(os.path.join(ORIG_ROOT, "train", "*")))]

# ----------------------------------------------------------------------------
# 2. Define Augmentations
# ----------------------------------------------------------------------------
# GPU-based augmentations (Kornia)
kornia_transforms = torch.nn.Sequential(
    K.ColorJitter(brightness=0.25, contrast=0.15, saturation=0, hue=0, p=0.8),
    K.RandomGaussianNoise(mean=0.0, std=0.03, p=0.5),
    K.RandomHorizontalFlip(p=0.8),
    K.RandomVerticalFlip(p=0.2),
    K.RandomRotation(degrees=30.0, p=0.3),
    K.RandomResizedCrop(size=TARGET_SIZE, scale=(0.9, 1.1), p=0.3),
    K.RandomPerspective(distortion_scale=0.2, p=0.4),
    K.RandomSharpness(sharpness=(0.0, 0.5), p=0.2),
    K.RandomErasing(scale=(0.02, 0.08), p=0.25),
    K.RandomErasing(scale=(0.02, 0.06), p=0.35),
).to(DEVICE)

# CPU-based image compression + weather (Albumentations)
transform_cpu = A.Compose([
    A.ImageCompression(p=1.0),
    A.OneOf([
        A.RandomRain(p=1.0),
        A.RandomFog(fog_coef_range=(0.02, 0.05), alpha_coef=0.1, p=1.0)
    ], p=0.4)
])

# ----------------------------------------------------------------------------
# 3. Helper for CPU transforms and saving
# ----------------------------------------------------------------------------
def cpu_transform_and_save(gpu_aug_np, cls, batch_idx, img_idx, aug_idx, dst_dir):
    # Convert back to BGR for Albumentations
    gpu_aug_bgr = cv2.cvtColor(gpu_aug_np, cv2.COLOR_RGB2BGR)
    # Apply CPU-based augmentation
    final = transform_cpu(image=gpu_aug_bgr)['image']
    # Construct output filename and save
    out_name = f"{cls}_{batch_idx + img_idx}_aug{aug_idx + 1}.jpg"
    cv2.imwrite(os.path.join(dst_dir, out_name), final)

# ----------------------------------------------------------------------------
# 4. Batch Augmentation Function with multithreading
# ----------------------------------------------------------------------------
def augment_split(split: str):
    """
    Processes a dataset split ('train' or 'val') in batches, applies GPU and CPU augmentations,
    and saves both original and augmented images to AUG_ROOT using multithreading for CPU steps.
    """
    for cls in CLASSES:
        src_dir = os.path.join(ORIG_ROOT, split, cls)
        dst_dir = os.path.join(AUG_ROOT, split, cls)
        os.makedirs(dst_dir, exist_ok=True)
        files = sorted(glob(os.path.join(src_dir, "*.*")))

        # Iterate batches with progress bar
        for batch_start in tqdm(range(0, len(files), BATCH_SIZE), desc=f"{split}/{cls}"):
            batch_files = files[batch_start: batch_start + BATCH_SIZE]
            originals = []

            # Load and store originals
            for file in batch_files:
                img_bgr = cv2.imread(file)
                if img_bgr is None:
                    continue  # skip unreadable files
                originals.append((file, img_bgr.copy()))

            # Prepare thread pool for CPU work
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []

                for i, (file, orig_bgr) in enumerate(originals):
                    basename = os.path.basename(file)
                    # Save original image
                    cv2.imwrite(os.path.join(dst_dir, basename), orig_bgr)

                    for n in range(NUM_AUGMENTS):
                        # --- GPU transforms ---
                        img_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
                        img_resized = cv2.resize(img_rgb, TARGET_SIZE)
                        tensor = TF.to_tensor(img_resized).unsqueeze(0).to(DEVICE)
                        gpu_aug = kornia_transforms(tensor)
                        gpu_aug_np = (gpu_aug.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

                        # Schedule CPU transform and saving in a worker thread
                        futures.append(
                            executor.submit(
                                cpu_transform_and_save,
                                gpu_aug_np, cls, batch_start, i, n, dst_dir
                            )
                        )

                # Ensure all CPU tasks complete before moving on
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error processing image in batch {batch_start}: {e}")

# ----------------------------------------------------------------------------
# 5. Main Entry Point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Clean up previous outputs
    if os.path.exists(AUG_ROOT):
        shutil.rmtree(AUG_ROOT)

    # Process both splits
    for sp in ["train", "val"]:
        augment_split(sp)

    print(f"\n✨ Batched augmented data created at '{AUG_ROOT}' on {DEVICE} with {MAX_WORKERS} threads ✨")
