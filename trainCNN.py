import os
import time
import copy
import csv
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix # <- MODIFIED IMPORT
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import gc

# ----------------- DYNAMIC OUTPUT DIRECTORY -----------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"CNNResult_Torch_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"📂 All results will be saved in: {output_dir}")
# -----------------------------------------------------------

# 🚀 DEVICE & HYPERPARAMETERS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# --- IMPORTANT: Update this path to your dataset location ---
data_dir = "augmented_dataset" 
# -----------------------------------------------------------

train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

IMG_WIDTH = 112
IMG_HEIGHT = 112
BATCH_SIZE = 32
target_epochs = 50
learning_rate = 5e-6
early_stopping_patience = 10

# 📂 DATA LOADING & TRANSFORMS
# Note: transforms.ToTensor() automatically scales images to [0.0, 1.0]
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {
    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, data_transforms['val'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
    'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
}

class_names = image_datasets['train'].classes
num_classes = len(class_names)
print(f"Detected classes: {class_names}")

# ----------------- DEEPER CNN MODEL DEFINITION (PYTORCH) -----------------
class DeeperCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeeperCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1: Input [batch, 3, 112, 112] -> Output [batch, 32, 56, 56]
            nn.Conv2d(3, 32, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 2: Input [batch, 32, 56, 56] -> Output [batch, 128, 28, 28]
            nn.Conv2d(32, 64, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=5, padding='same'), 
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            
            # Block 4: Input [batch, 128, 28, 28] -> Output [batch, 128, 14, 14]
            nn.Conv2d(128, 128, kernel_size=7, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.35),
        )
        
        # Flatten and Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = DeeperCNN(num_classes).to(device)
print(model)
# --------------------------------------------------------------------

# LOSS, OPTIMIZER, SCHEDULER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=4)

# TRAINING LOOP
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_acc = 0.0
best_wts = copy.deepcopy(model.state_dict())
epochs_no_improve = 0

all_val_labels = np.array([])
all_val_probs = None

for epoch in range(target_epochs):
    print(f"\n🔥 Epoch {epoch+1}/{target_epochs}")
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0
        
        epoch_labels_list = []
        epoch_probs_list = []

        loop = tqdm(dataloaders[phase], desc=f"{phase.capitalize():<5} [{epoch+1}/{target_epochs}]", leave=True)
        for inputs, labels in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if phase == 'val':
                probs = torch.softmax(outputs, 1).detach().cpu()
                epoch_probs_list.append(probs)
                epoch_labels_list.append(labels.detach().cpu())

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])
        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        if phase == 'train':
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc.item())
        else: # Validation phase
            history['val_loss'].append(epoch_loss)
            history['val_acc'].append(epoch_acc.item())
            scheduler.step(epoch_acc)

            if epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())
                model_path = os.path.join(output_dir, "Deeper_CNN_Torch_Best.pt")
                torch.save(model.state_dict(), model_path)
                print(f"✅ New best model saved to {model_path}")
                epochs_no_improve = 0
                # Save the predictions from the best epoch
                all_val_labels = torch.cat(epoch_labels_list).numpy()
                all_val_probs = torch.cat(epoch_probs_list).numpy()
            else:
                epochs_no_improve += 1
    
    if epochs_no_improve >= early_stopping_patience:
        print(f"\n✋ Early stopping triggered after {epoch+1} epochs.")
        break
    
    gc.collect()
    torch.cuda.empty_cache()

print(f"\n🎉 Best Validation Accuracy: {best_val_acc:.4f}")
model.load_state_dict(best_wts)

# --- PUBLICATION-QUALITY GRAPHS & RESULTS ---
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
print(f"📈 Combined metrics plot saved to {metrics_plot_path}")

# 2. Generate All Other Reports if Validation Data Exists
if all_val_labels.size > 0:
    # Get predicted class labels by finding the index with the maximum probability
    all_val_preds = np.argmax(all_val_probs, axis=1)

    # --- ROC/AUC PLOT ---
    y_true_bin = label_binarize(all_val_labels, classes=list(range(num_classes)))
    fpr, tpr, roc_auc = {}, {}, {}
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", num_classes)

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
    roc_plot_path = os.path.join(output_dir, "roc_curves_all_classes.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"📈 All-in-one ROC curve plot saved to {roc_plot_path}")

    # --- SAVE AUC SCORES TO CSV ---
    auc_csv_path = os.path.join(output_dir, "auc_scores.csv")
    with open(auc_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Class', 'AUC'])
        for i, cname in enumerate(class_names):
            writer.writerow([cname, f"{roc_auc[i]:.4f}"])
    print(f"📄 AUC scores saved to {auc_csv_path}")

    # --- CLASSIFICATION REPORT ---
    report_str = classification_report(
        all_val_labels,
        all_val_preds,
        target_names=class_names,
        digits=2
    )
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=====================\n\n")
        f.write(report_str)
    print(f"📄 Classification report saved to {report_path}")

    # --- NEW: CONFUSION MATRIX ---
    cm = confusion_matrix(all_val_labels, all_val_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
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

else:
    print("⚠️ Could not generate reports because no validation predictions were saved.")

print("\n🎉 Training complete!")