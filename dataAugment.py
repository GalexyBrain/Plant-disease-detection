import os
import shutil
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import albumentations as A
import cv2

# ----------------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------------

# Number of augmented copies per input image
NUM_AUGMENTS = 3

# Paths
ORIG_ROOT = "dataset"
AUG_ROOT = "augmented_dataset"

# Define classes: infer from train folder
CLASSES = [
    os.path.basename(d)
    for d in sorted(glob(os.path.join(ORIG_ROOT, "train", "*")))
]

# ----------------------------------------------------------------------------
# 2. Augmentation pipeline
# ----------------------------------------------------------------------------

transform = A.Compose([
    # Noise & Compression
    A.ImageCompression(p=1),  # simulate JPEG artifacts

    # Photometric transforms
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.CLAHE(p=0.3),                              # adaptive histogram equalization

    # Blur
    A.OneOf([
        A.MotionBlur(blur_limit=5, p=1.0),
        A.MedianBlur(blur_limit=3, p=1.0),
    ], p=0.35),

    # Geometric transforms
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomScale(scale_limit=0.1, p=0.3),
    A.Perspective(p=0.4),
    A.ElasticTransform(p=0.3),
    A.GridDistortion(p=0.3),

    # Occlusion
    A.CoarseDropout(p=0.3),

    # Weather effects
    A.OneOf([
        A.RandomRain(p=1.0),
        A.RandomFog(p=1.0)
    ], p=0.4)
])

# ----------------------------------------------------------------------------
# 3. Task builder and worker
# ----------------------------------------------------------------------------

def build_tasks(split: str):
    """
    Build a list of tasks for ThreadPoolExecutor. Each task is a tuple:
    (img_path, split, cls, idx_in_class)
    """
    tasks = []
    for cls in CLASSES:
        src_dir = os.path.join(ORIG_ROOT, split, cls)
        images = sorted(glob(os.path.join(src_dir, "*.*")))
        for idx, img_path in enumerate(images):
            tasks.append((img_path, split, cls, idx))
    return tasks


def worker(task):
    img_path, split, cls, idx = task
    # prepare destination directory
    dst_dir = os.path.join(AUG_ROOT, split, cls)
    os.makedirs(dst_dir, exist_ok=True)

    # copy original
    filename = os.path.basename(img_path)
    shutil.copy(img_path, os.path.join(dst_dir, filename))

    # read image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # generate augments
    base_name, ext = os.path.splitext(filename)
    for i in range(NUM_AUGMENTS):
        augmented = transform(image=image)["image"]
        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        aug_name = f"{cls}_{idx}_aug{i+1}{ext}"
        cv2.imwrite(os.path.join(dst_dir, aug_name), augmented)

    return True

# ----------------------------------------------------------------------------
# 4. Run multithreaded
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # Clean target if exists
    if os.path.exists(AUG_ROOT):
        shutil.rmtree(AUG_ROOT)

    for split in ["train", "val"]:
        # build tasks
        tasks = build_tasks(split)
        # ensure destination folders exist
        for cls in CLASSES:
            os.makedirs(os.path.join(AUG_ROOT, split, cls), exist_ok=True)

        # process with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            list(tqdm(executor.map(worker, tasks), total=len(tasks), desc=split))

    print(f"\n✨ Multithreaded augmented dataset created at '{AUG_ROOT}' ✨")
