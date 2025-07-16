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
from torchvision import datasets, transforms
from torchvision.models.vision_transformer import VisionTransformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import csv
from tqdm import tqdm
import gc


# 🚀 Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)

torch.set_float32_matmul_precision('high')


# Hyperparameters
num_epochs = 25
batch_size = 16
learning_rate = 5e-5
early_stopping = 3

early_stopping_patience = early_stopping

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
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Datasets & optimized DataLoaders
train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
val_ds = datasets.ImageFolder(val_dir, transform=val_transform)
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=8, prefetch_factor=2, persistent_workers=True,
    pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False,
    num_workers=8, prefetch_factor=2, persistent_workers=True,
    pin_memory=True
)

class_names = train_ds.classes
num_labels = len(class_names)
print("📂 Classes:", class_names)

# Model definition
model = VisionTransformer(
    image_size=224,
    patch_size=16,
    num_layers=12,
    num_heads=12,
    hidden_dim=768,
    mlp_dim=3072,
    num_classes=num_labels
).to(device)

# Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=3)

# Metrics storage
train_losses, val_losses = [], []
train_accs, val_accs = [], []
all_val_labels, all_val_probs = [], []

# Best model tracking
best_val_acc = 0.0
best_wts = copy.deepcopy(model.state_dict())

# Training Loop
try:
    for epoch in range(num_epochs):
        print(f"\n🔥 Epoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val']:
            model.train() if phase=='train' else model.eval()
            loader = train_loader if phase=='train' else val_loader
            running_loss, running_corrects = 0.0, 0
            epoch_labels, epoch_probs = [], []

            loop = tqdm(loader, desc=f"{phase.capitalize()} [{epoch+1}/{num_epochs}]", leave=False, ncols=100)
            for inputs, labels in loop:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds==labels.data).item()
                if phase=='val':
                    probs = torch.softmax(outputs, 1).detach().cpu().numpy()
                    epoch_probs.append(probs)
                    epoch_labels.append(labels.detach().cpu().numpy())
                current_loss = running_loss / ((loop.n+1)*batch_size)
                current_acc = running_corrects / ((loop.n+1)*batch_size)
                loop.set_postfix({'Loss':f"{current_loss:.4f}",'Acc':f"{current_acc*100:.2f}%"})

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects / len(loader.dataset)
            if phase=='train':
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
                    print("✅ Model checkpoint saved!")
                    early_stopping_patience = early_stopping
                else:
                    early_stopping_patience -= 1
                if early_stopping_patience == 0:
                    raise KeyboardInterrupt
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.2f}%")
        
        torch.cuda.empty_cache()
        gc.collect()
        all_val_probs.clear()
        all_val_labels.clear()
        time.sleep(0.5)
except Exception as ex:
    print("Training interrupted ... finish training and plot graphs")

print(f"\n🎉 Best Validation Accuracy: {best_val_acc*100:.2f}%")
model.load_state_dict(best_wts)
torch.save(best_wts, "final_vit_model.pt")
print("💾 Final model saved as final_vit_model.pt")

# Plot Loss & Accuracy
epochs = range(1, len(train_losses) + 1)
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

# ROC–AUC per class and CSV
y_true = np.concatenate(all_val_labels)
y_score = np.vstack(all_val_probs)
y_true_bin = label_binarize(y_true, classes=list(range(num_labels)))
fpr, tpr, roc_auc = {}, {}, {}
for i, cname in enumerate(class_names):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    plt.plot(fpr[i], tpr[i], label=f'{cname} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.title(f'ROC Curve: {cname}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='lower right')
    plt.savefig(f"roc_curve_{cname.replace(' ','_')}.png")
    plt.close()

with open('auc_scores.csv','w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Class','AUC'])
    for cname, score in roc_auc.items():
        writer.writerow([class_names[cname], f"{score:.4f}"])

plt.figure()
plt.bar(class_names, [roc_auc[i] for i in range(num_labels)])
plt.title('AUC Scores by Class')
plt.ylabel('AUC')
plt.ylim([0,1])
plt.savefig('auc_bar_plot.png')
plt.close()

print("📈 ROC curves, AUC CSV, and AUC bar plot saved.")
