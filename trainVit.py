import os
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
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
import csv
from tqdm import tqdm
import gc
import seaborn as sns
from datetime import datetime

# ----------------- DYNAMIC OUTPUT DIRECTORY -----------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"results_vit_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"📂 All results will be saved in: {output_dir}")
# -----------------------------------------------------------

# 🚀 Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)

# NEW: Enable TensorFloat32 for better performance on compatible GPUs
torch.set_float32_matmul_precision('high')

# --- Hyperparameters ---
num_epochs = 50
batch_size = 16
# NEW: Gradient accumulation steps
accumulation_steps = 4 # Effective batch size will be batch_size * accumulation_steps = 32
learning_rate = 1e-4
scheduler_patience = 3
early_stopping_patience = 10

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
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True, persistent_workers=True)

class_names = train_ds.classes
num_labels = len(class_names)
print("📂 Classes:", class_names)

# Load Vision Transformer (ViT) from scratch
model = models.vit_b_16(weights=None)
num_ftrs = model.heads.head.in_features
model.heads.head = nn.Linear(num_ftrs, num_labels)
model = model.to(device)
# Apply torch.compile for a significant performance boost
model = torch.compile(model)
print("✅ Model compiled for performance.")

# Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=scheduler_patience)

# --- Training Loop with AMP and Gradient Accumulation ---
# NEW: Initialize GradScaler for mixed precision
scaler = torch.amp.GradScaler('cuda')

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_acc = 0.0
best_wts = copy.deepcopy(model.state_dict())
all_val_labels, all_val_probs = None, None
epochs_no_improve = 0

for epoch in range(num_epochs):
    print(f"\n🔥 Epoch {epoch+1}/{num_epochs}")
    for phase in ['train', 'val']:
        model.train() if phase == 'train' else model.eval()
        loader = train_loader if phase == 'train' else val_loader
        running_loss, running_corrects = 0.0, 0
        epoch_labels, epoch_probs_list = [], []

        loop = tqdm(loader, desc=f"{phase.capitalize():<5} [{epoch+1}/{num_epochs}]", leave=True, ncols=100)
        
        # MODIFIED: Loop structure for gradient accumulation
        for i, (inputs, labels) in enumerate(loop):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Use autocast for the forward pass (mixed precision)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # Normalize loss for accumulation
                if phase == 'train':
                    loss = loss / accumulation_steps

            if phase == 'train':
                # Scale loss and call backward() to create scaled gradients
                scaler.scale(loss).backward()
                
                # Update weights only every accumulation_steps
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            
            # --- Metrics Calculation (Unaffected) ---
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0) * (accumulation_steps if phase == 'train' else 1)
            running_corrects += torch.sum(preds == labels.data).item()

            if phase == 'val':
                probs = torch.softmax(outputs, 1).detach().cpu()
                epoch_probs_list.append(probs)
                epoch_labels.append(labels.detach().cpu())

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = running_corrects / len(loader.dataset)

        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.2f}%")

        if phase == 'train':
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
        else: # Validation phase
            history['val_loss'].append(epoch_loss)
            history['val_acc'].append(epoch_acc)
            scheduler.step(epoch_acc)

            if epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                # Use _orig_mod to get the state_dict of the uncompiled model
                best_wts = copy.deepcopy(model._orig_mod.state_dict())
                best_model_path = os.path.join(output_dir, "best_vit_model.pt")
                torch.save(best_wts, best_model_path)
                print(f"✅ New best model saved to {best_model_path}")
                all_val_labels = torch.cat(epoch_labels).numpy()
                all_val_probs = torch.cat(epoch_probs_list).numpy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

    if epochs_no_improve >= early_stopping_patience:
        print(f"\n✋ Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
        break

    torch.cuda.empty_cache()
    gc.collect()

print(f"\n🎉 Best Validation Accuracy: {best_val_acc*100:.2f}%")
# Load the best weights into the uncompiled model for saving and analysis
model._orig_mod.load_state_dict(best_wts)
final_model_path = os.path.join(output_dir, "final_vit_model.pt")
torch.save(best_wts, final_model_path)
print(f"💾 Final best model saved as {final_model_path}")

# --- Publication-Quality Graphs & Results ---
# (This entire section remains the same as it uses the final saved results)
sns.set_theme(style="whitegrid")

# 1. Combined Loss and Accuracy Plot
if history['train_loss']:
    num_actual_epochs = len(history['train_loss'])
    epochs_range = range(1, num_actual_epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss')
    ax1.plot(epochs_range, history['val_loss'], 'r-o', label='Validation Loss')
    ax1.set_title('Model Loss', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True)

    ax2.plot(epochs_range, history['train_acc'], 'b-o', label='Train Accuracy')
    ax2.plot(epochs_range, history['val_acc'], 'r-o', label='Validation Accuracy')
    ax2.set_title('Model Accuracy', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True)

    fig.suptitle('Training and Validation Metrics', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    metrics_plot_path = os.path.join(output_dir, "loss_acc_curves_combined.png")
    plt.savefig(metrics_plot_path, dpi=300)
    plt.close()
    print(f"\n📈 Combined metrics plot saved to {metrics_plot_path}")

# Generate other reports only if validation was performed and a best model was found
if all_val_labels is not None and all_val_probs is not None:
    all_val_preds = np.argmax(all_val_probs, axis=1)

    # 2. Confusion Matrix
    cm = confusion_matrix(all_val_labels, all_val_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"📊 Confusion matrix saved to {cm_path}")
    
    # 3. All-in-One ROC-AUC Plot
    y_true_bin = label_binarize(all_val_labels, classes=list(range(num_labels)))
    fpr, tpr, roc_auc = {}, {}, {}
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", num_labels)
    for i, (cname, color) in enumerate(zip(class_names, colors)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_val_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC for {cname} (AUC = {roc_auc[i]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='No-Skill Line')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Multi-Class ROC Curves', fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    roc_plot_path = os.path.join(output_dir, "roc_curves_all_classes.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"📈 All-in-one ROC curve plot saved to {roc_plot_path}")

    # 4. Save AUC scores to CSV
    csv_path = os.path.join(output_dir, "auc_scores.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Class', 'AUC'])
        for i, cname in enumerate(class_names):
            writer.writerow([cname, f"{roc_auc[i]:.4f}"])
    print(f"📄 AUC scores saved to {csv_path}")
    
    # 5. Classification Report
    report_str = classification_report(all_val_labels, all_val_preds, target_names=class_names, digits=2)
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=====================\n\n")
        f.write(report_str)
    print(f"📄 Classification report saved to {report_path}")

else:
    print("\n⚠️ Could not generate reports because no validation predictions were saved from a best epoch.")

print("\nAll tasks completed.")