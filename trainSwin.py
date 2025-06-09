import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, SwinForImageClassification
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 📁 Output Directory
output_dir = "./SwinOutput"
os.makedirs(output_dir, exist_ok=True)

# ⚙️ Configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

num_epochs = 6
learning_rate = 5e-5
batch_size = 2
model_name = "microsoft/swin-tiny-patch4-window7-224"

# 📊 Data Augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 📂 Data Paths
data_dir = "./dataset"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

class_names = train_dataset.classes
num_labels = len(class_names)
print("Detected Classes:", class_names)

# 🧠 Load Swin Transformer
feature_extractor = AutoImageProcessor.from_pretrained(model_name)
model = SwinForImageClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)
model.to(device)

# 🥺 Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=2)

best_model_wts = copy.deepcopy(model.state_dict())
best_val_acc = 0.0

# 📊 Logs for plotting
train_acc_log, val_acc_log = [], []
train_loss_log, val_loss_log = [], []

# 🔁 Training Loop
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 30)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            dataloader = train_loader
        else:
            model.eval()
            dataloader = val_loader

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs).logits
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

        if phase == 'train':
            train_acc_log.append(epoch_acc.item())
            train_loss_log.append(epoch_loss)
        else:
            val_acc_log.append(epoch_acc.item())
            val_loss_log.append(epoch_loss)
            scheduler.step(epoch_acc)
            if epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(output_dir, "best_swin_model.pt"))
                print(">> 🧠 Best model saved!")
    torch.cuda.empty_cache()



# 📆 Save Final Model
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), os.path.join(output_dir, "final_swin_model.pt"))
print(f"\n🌟 Best Validation Accuracy: {best_val_acc:.4f}")
print("✅ Final model saved as final_swin_model.pt")

# 📊 Plot accuracy and loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_acc_log, label='Train Acc')
plt.plot(val_acc_log, label='Val Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss_log, label='Train Loss')
plt.plot(val_loss_log, label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "training_metrics.png"))
print("🔺 Accuracy & Loss plot saved as training_metrics.png")

