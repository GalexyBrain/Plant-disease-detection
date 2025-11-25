import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import csv

# Verbosity
verbose = True

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters
model_name = "google/vit-base-patch16-224"
num_epochs = 6
batch_size = 16
learning_rate = 5e-5

data_dir = "dataset"
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

# Datasets & Loaders
train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
val_ds = datasets.ImageFolder(val_dir, transform=val_transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

class_names = train_ds.classes
num_labels = len(class_names)
print("Classes:", class_names)

# Model & Feature Extractor
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(
    model_name, num_labels=num_labels, ignore_mismatched_sizes=True
)
model.to(device)

# Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=3)

# Metrics storage
train_losses, val_losses = [], []
train_accs, val_accs = [], []
all_val_labels, all_val_probs = [], []

# Best model
best_val_acc = 0.0
best_wts = copy.deepcopy(model.state_dict())

# Training Loop
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            loader = train_loader
        else:
            model.eval()
            loader = val_loader
            epoch_labels, epoch_probs = [], []

        running_loss, running_corrects = 0.0, 0

        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs).logits
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels.data).sum().item()

            if verbose:
                print(f"[{phase}] Epoch {epoch+1} Batch {i+1}/{len(loader)} Loss: {loss.item():.4f}")

            if phase == 'val':
                probs = torch.softmax(outputs, 1).detach().cpu().numpy()
                epoch_probs.append(probs)
                epoch_labels.append(labels.detach().cpu().numpy())

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = running_corrects / len(loader.dataset)

        if phase == 'train':
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)
        else:
            val_losses.append(epoch_loss)
            val_accs.append(epoch_acc)
            all_val_probs.append(np.vstack(epoch_probs))
            all_val_labels.append(np.concatenate(epoch_labels))
            scheduler.step(epoch_acc)

            if epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())
                torch.save(best_wts, "best_vit_model.pt")
                print(">> Model checkpoint saved!")

        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
model.load_state_dict(best_wts)
torch.save(best_wts, "final_vit_model.pt")
print("Final model saved as final_vit_model.pt")

# Loss & Acc Curves
epochs = range(1, num_epochs + 1)
plt.figure()
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.plot(epochs, train_accs, label='Train Acc')
plt.plot(epochs, val_accs, label='Val Acc')
plt.title('Loss & Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('loss_acc_curve.png')
plt.close()

# ROC–AUC
y_true = np.concatenate(all_val_labels)
y_score = np.vstack(all_val_probs)
y_true_bin = label_binarize(y_true, classes=list(range(num_labels)))
fpr, tpr, roc_auc = {}, {}, {}
for i, cname in enumerate(class_names):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    plt.plot(fpr[i], tpr[i], label=f'{cname} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve: {cname}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    filename = f"roc_curve_{cname.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()

# Save AUC CSV
with open('auc_scores.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Class', 'AUC'])
    for cname, score in roc_auc.items():
        writer.writerow([class_names[cname], f"{score:.4f}"])

# Optional: AUC bar plot
plt.figure()
plt.bar(class_names, [roc_auc[i] for i in range(num_labels)])
plt.title('AUC Scores by Class')
plt.ylabel('AUC')
plt.ylim([0, 1])
plt.savefig('auc_bar_plot.png')
plt.close()

print("ROC curves, AUC CSV, and AUC bar plot saved.")
