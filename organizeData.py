import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
dataset_dir = './PlantVillage'  # Adjust if necessary
output_dir = 'dataset'
train_ratio = 0.8  # 80% training, 20% validation

# Create train and val directories
for split in ['train', 'val']:
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

# Split data
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    images = os.listdir(class_path)
    train_images, val_images = train_test_split(images, train_size=train_ratio, random_state=42)

    for image in train_images:
        src = os.path.join(class_path, image)
        dst = os.path.join(output_dir, 'train', class_name, image)
        shutil.copyfile(src, dst)

    for image in val_images:
        src = os.path.join(class_path, image)
        dst = os.path.join(output_dir, 'val', class_name, image)
        shutil.copyfile(src, dst)

print("Dataset organized into training and validation sets.")
