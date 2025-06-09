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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define hyperparameters
model_name = "google/vit-base-patch16-224"
num_epochs = 6
batch_size = 16
learning_rate = 5e-5

# Improved data augmentation and normalization
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # randomly rotate by ±15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define directories (assumes structure: dataset/train and dataset/val)
data_dir = "dataset"  
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

# Create datasets and dataloaders
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

class_names = train_dataset.classes
num_labels = len(class_names)
print("Classes:", class_names)

# Load pre-trained feature extractor (can be used during inference)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Load pre-trained ViT model and adjust classification head
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    ignore_mismatched_sizes=True  # allows adapting head size
)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
# Use a scheduler that reduces LR when validation accuracy plateaus
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=3)

# Training loop with checkpointing (saves best model based on validation accuracy)
best_model_wts = copy.deepcopy(model.state_dict())
best_val_acc = 0.0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 30)
    
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # set model to training mode
            dataloader = train_loader
        else:
            model.eval()   # set model to evaluation mode
            dataloader = val_loader
        
        running_loss = 0.0
        running_corrects = 0
        
        # Iterate over data.
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass; track gradients only in training phase
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs).logits
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                # Backward pass and optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)
        
        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
        
        # Step the scheduler with validation accuracy
        if phase == 'val':
            scheduler.step(epoch_acc)
            # Checkpoint: save model if validation accuracy improved
            if epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "best_vit_model.pt")
                print(">> Model checkpoint saved!")
                
print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

# Load best model weights and save final model
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), "final_vit_model.pt")
print("Final model saved as final_vit_model.pt")
