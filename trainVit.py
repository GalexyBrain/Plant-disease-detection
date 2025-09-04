import os
import time
import copy
import csv
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
import seaborn as sns

print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")

# --- Configuration ---
# Set the path to your main data directory
data_dir = 'dataset'
# Directory to save results with a timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = f'ViT_Results_{timestamp}'
os.makedirs(results_dir, exist_ok=True)
print(f"📂 All results will be saved in: {results_dir}")

# Model parameters
NUM_CLASSES = 9
BATCH_SIZE = 16
NUM_EPOCHS = 5
# A common learning rate for fine-tuning ViT
LEARNING_RATE = 2e-5

# --- Data Loading and Transformation ---
# ViT-B/16 was trained on 224x224 images, so these transforms are appropriate.
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
print(f"Found {len(class_names)} classes: {class_names}")


# --- Model Definition (Fine-tuning Pretrained ViT) ---
# Load the pretrained ViT-Base-Patch16-224 model
model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

# The classifier head in ViT is in the 'heads' attribute
# It's a single nn.Linear layer named 'head'
num_ftrs = model.heads.head.in_features
# Replace the classifier head with a new one for our number of classes
model.heads.head = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(device)
print("✅ Model setup complete. Fine-tuning a pretrained Vision Transformer.")

# --- Training Setup ---
criterion = nn.CrossEntropyLoss()
# We will optimize all parameters, but with a small learning rate for fine-tuning
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training and Validation Function ---
def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # To store predictions from the best epoch for detailed analysis
    all_val_labels = None
    all_val_probs = None

    for epoch in range(num_epochs):
        print(f'🔥 Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Lists to store labels and probabilities for the current validation epoch
            epoch_labels_list = []
            epoch_probs_list = []

            # Iterate over data using a progress bar
            progress_bar = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Epoch {epoch+1}/{num_epochs}')
            
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # In validation phase, collect all predictions for later analysis
                if phase == 'val':
                    # Get probabilities by applying softmax to the output logits
                    probs = torch.softmax(outputs, 1).detach().cpu()
                    epoch_probs_list.append(probs)
                    epoch_labels_list.append(labels.detach().cpu())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Record the loss and accuracy for plotting
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            # Deep copy the model if it's the best one so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"✅ New best model found with validation accuracy: {best_acc:.4f}")
                # Save the predictions from this best epoch
                all_val_labels = torch.cat(epoch_labels_list).numpy()
                all_val_probs = torch.cat(epoch_probs_list).numpy()

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'🎉 Best val Acc: {best_acc:4f}')

    # Load best model weights before returning
    model.load_state_dict(best_model_wts)
    return model, history, all_val_labels, all_val_probs

# --- Execute Training ---
print("Starting training...")
trained_model, history, all_val_labels, all_val_probs = train_model(model, criterion, optimizer, num_epochs=NUM_EPOCHS)

# --- Save Final Model ---
print("\nSaving the best model...")
model_save_path = os.path.join(results_dir, 'vit_b_16_best_weights.pth')
torch.save(trained_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# --- Generate All Plots and Reports ---
print("\nGenerating and saving plots and reports...")
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
metrics_plot_path = os.path.join(results_dir, "loss_acc_curves_combined.png")
plt.savefig(metrics_plot_path, dpi=300)
plt.close()
print(f"📈 Combined metrics plot saved to {metrics_plot_path}")

# Generate other reports only if validation was performed and results were saved
if all_val_labels is not None and all_val_probs is not None:
    # Get predicted class indices from the probabilities
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
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"📊 Confusion matrix saved to {cm_path}")

    # 3. ROC Curve and AUC Scores (One-vs-Rest)
    # Binarize the true labels for multi-class ROC analysis
    y_true_bin = label_binarize(all_val_labels, classes=range(NUM_CLASSES))
    fpr, tpr, roc_auc = {}, {}, {}
    
    plt.figure(figsize=(12, 10))
    colors = sns.color_palette("husl", NUM_CLASSES)
    
    for i, color in enumerate(colors):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_val_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC for {class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='No-Skill Line')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Multi-Class Receiver Operating Characteristic (ROC) Curves', fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    roc_plot_path = os.path.join(results_dir, "roc_curves_all_classes.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"📈 All-in-one ROC curve plot saved to {roc_plot_path}")

    # 4. Save AUC scores to CSV
    auc_csv_path = os.path.join(results_dir, "auc_scores.csv")
    with open(auc_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Class', 'AUC'])
        for i, cname in enumerate(class_names):
            writer.writerow([cname, f"{roc_auc[i]:.4f}"])
    print(f"📄 AUC scores saved to {auc_csv_path}")

    # 5. Classification Report
    report_str = classification_report(all_val_labels, all_val_preds, target_names=class_names, digits=3)
    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=====================\n\n")
        f.write(report_str)
    print(f"📄 Classification report saved to {report_path}")

else:
    print("⚠️ Could not generate detailed reports because no validation predictions were saved from a best epoch.")

print("\nAll tasks completed. ✅")