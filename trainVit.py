import os
# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import csv
from tqdm import tqdm
import gc
import seaborn as sns
# NEW: Import datetime to create timestamped directories
from datetime import datetime

# ----------------- NEW SECTION: DYNAMIC OUTPUT DIRECTORY -----------------
# 1. Create a unique directory name using the current timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"results_vit_{timestamp}"

# 2. Create the directory
os.makedirs(output_dir, exist_ok=True)
print(f"📂 All results will be saved in: {output_dir}")
# --------------------------------------------------------------------------

# 🚀 Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)

torch.set_float32_matmul_precision('high')

# Hyperparameters
num_epochs = 6
batch_size = 16
learning_rate = 5e-5
early_stopping_patience = 3 

# Data directories
data_dir = "augmented_dataset"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets & optimized DataLoaders
train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
val_ds = datasets.ImageFolder(val_dir, transform=val_transform)
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=8, prefetch_factor=2, persistent_workers=True, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False,
    num_workers=8, prefetch_factor=2, persistent_workers=True, pin_memory=True
)

class_names = train_ds.classes
num_labels = len(class_names)
print("📂 Classes:", class_names)

# Load the pre-trained Vision Transformer (ViT)
model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
num_ftrs = model.heads.head.in_features
model.heads.head = nn.Linear(num_ftrs, num_labels)
model = model.to(device)

# Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=3)

# Metrics storage
history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': []
}

# Best model tracking
best_val_acc = 0.0
best_wts = copy.deepcopy(model.state_dict())

# Early stopping setup
epochs_no_improve = 0
early_stop_triggered = False

# Training Loop
for epoch in range(num_epochs):
    print(f"\n🔥 Epoch {epoch+1}/{num_epochs}")
    for phase in ['train', 'val']:
        model.train() if phase == 'train' else model.eval()
        loader = train_loader if phase == 'train' else val_loader
        running_loss, running_corrects = 0.0, 0
        
        epoch_labels, epoch_probs = [], []

        loop = tqdm(loader, desc=f"{phase.capitalize():<5} [{epoch+1}/{num_epochs}]", leave=False, ncols=100)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            
            if phase == 'val':
                probs = torch.softmax(outputs, 1).detach().cpu()
                epoch_probs.append(probs)
                epoch_labels.append(labels.detach().cpu())

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = running_corrects / len(loader.dataset)
        
        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.2f}%")

        if phase == 'train':
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
        else:
            history['val_loss'].append(epoch_loss)
            history['val_acc'].append(epoch_acc)
            scheduler.step(epoch_acc)

            if epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())
                # MODIFIED: Save model to the new directory
                best_model_path = os.path.join(output_dir, "best_vit_model.pt")
                torch.save(best_wts, best_model_path)
                print(f"✅ New best model saved to {best_model_path}")
                all_val_labels = torch.cat(epoch_labels).numpy()
                all_val_probs = torch.cat(epoch_probs).numpy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

    if epochs_no_improve >= early_stopping_patience:
        print(f"\n✋ Early stopping triggered after {epoch+1} epochs.")
        early_stop_triggered = True
        break
    
    torch.cuda.empty_cache()
    gc.collect()

print(f"\n🎉 Best Validation Accuracy: {best_val_acc*100:.2f}%")
model.load_state_dict(best_wts)
# MODIFIED: Save final model to the new directory
final_model_path = os.path.join(output_dir, "final_vit_model.pt")
torch.save(best_wts, final_model_path)
print(f"💾 Final best model saved as {final_model_path}")

# --- Publication-Quality Graphs & Results (Now saved to output_dir) ---

sns.set_theme(style="whitegrid")

# 1. Combined Loss and Accuracy Plot
num_actual_epochs = len(history['train_loss'])
epochs_range = range(1, num_actual_epochs + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss')
ax1.plot(epochs_range, history['val_loss'], 'r-o', label='Validation Loss')
ax1.set_title('Model Loss', fontsize=16)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.legend(fontsize=12)

ax2.plot(epochs_range, history['train_acc'], 'b-o', label='Train Accuracy')
ax2.plot(epochs_range, history['val_acc'], 'r-o', label='Validation Accuracy')
ax2.set_title('Model Accuracy', fontsize=16)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.legend(fontsize=12)

fig.suptitle('Training and Validation Metrics', fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# MODIFIED: Save plot to the new directory
metrics_plot_path = os.path.join(output_dir, "loss_acc_curves_combined.png")
plt.savefig(metrics_plot_path, dpi=300)
plt.close()
print(f"📈 Combined metrics plot saved to {metrics_plot_path}")

# 2. All-in-One ROC-AUC Plot
y_true_bin = label_binarize(all_val_labels, classes=list(range(num_labels)))
fpr, tpr, roc_auc = {}, {}, {}
plt.figure(figsize=(10, 8))
colors = sns.color_palette("husl", num_labels)

for i, (cname, color) in enumerate(zip(class_names, colors)):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_val_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve for {cname} (AUC = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='No-Skill Line')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('Multi-Class Receiver Operating Characteristic (ROC) Curves', fontsize=16)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True)
# MODIFIED: Save plot to the new directory
roc_plot_path = os.path.join(output_dir, "roc_curves_all_classes.png")
plt.savefig(roc_plot_path, dpi=300)
plt.close()
print(f"📈 All-in-one ROC curve plot saved to {roc_plot_path}")

# 3. Save AUC scores to CSV
# MODIFIED: Save CSV to the new directory
csv_path = os.path.join(output_dir, "auc_scores.csv")
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Class', 'AUC'])
    for i, cname in enumerate(class_names):
        writer.writerow([cname, f"{roc_auc[i]:.4f}"])

print(f"📄 AUC scores saved to {csv_path}")