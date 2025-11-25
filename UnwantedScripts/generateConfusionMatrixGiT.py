#!/usr/bin/env python
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define validation transforms (should match those used during training)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define directories (assumes dataset/val contains the validation images)
data_dir = "dataset"
val_dir = os.path.join(data_dir, "val")

# Create validation dataset and dataloader
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# Retrieve class names
class_names = val_dataset.classes
num_labels = len(class_names)
print("Classes:", class_names)

# Load the pre-trained ViT model (adjust head size as needed)
model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)
# Load saved model weights (final model checkpoint)
model.load_state_dict(torch.load("final_vit_model.pt"))
model.to(device)
model.eval()

# Store predictions and ground truth labels
all_preds = []
all_labels = []

# Run inference over the validation dataset
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs).logits
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute confusion matrix using scikit-learn
cm = confusion_matrix(all_labels, all_preds)

# Plot the confusion matrix using seaborn's heatmap
plt.figure(figsize=(10, 8))  # Bigger figure to give labels space!
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.tight_layout()  # ✅ Prevent clipping
plt.savefig("confusion_matrix.png")

print("Confusion matrix saved as confusion_matrix.png")


