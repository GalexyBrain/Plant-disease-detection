import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
import os
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns

print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")

# --- Configuration ---
# Set the path to your main data directory
data_dir = 'augmented_dataset'
# Directory to save results
results_dir = 'ResNet_Results'
os.makedirs(results_dir, exist_ok=True)


# Model parameters
NUM_CLASSES = 9
BATCH_SIZE = 128
NUM_EPOCHS = 30
LEARNING_RATE = 0.00001

# --- Data Loading and Transformation ---
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
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Found {len(class_names)} classes: {class_names}")


# --- Model Definition ---
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(device)
print("Model setup complete.")

# --- Training Setup ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# --- Training and Validation Function ---
def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()
    
    # History for plotting
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Use tqdm for a progress bar
            progress_bar = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Epoch {epoch+1}/{num_epochs}')
            
            for inputs, labels in progress_bar:
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

                # Update progress bar description
                progress_bar.set_postfix(loss=running_loss/((progress_bar.n + 1) * BATCH_SIZE), acc=running_corrects.double()/((progress_bar.n + 1) * BATCH_SIZE))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Record loss and accuracy for plotting
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history

# --- Plotting Functions ---
def plot_metrics(history, results_dir):
    """Plots and saves the training/validation accuracy and loss."""
    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'accuracy_vs_epochs.png'))
    plt.close()

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'loss_vs_epochs.png'))
    plt.close()
    print("Accuracy and Loss plots saved.")

def plot_confusion_matrix_and_roc(model, dataloader, device, class_names, results_dir):
    """Plots and saves the confusion matrix and ROC curves."""
    model.eval()
    y_true = []
    y_pred = []
    y_score = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            # Use softmax to get probabilities for ROC curve
            probabilities = F.softmax(outputs, dim=1)
            y_score.extend(probabilities.cpu().numpy())

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()
    print("Confusion Matrix saved.")

    # --- ROC Curve and AUC ---
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    y_score = np.array(y_score)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(12, 10))
    colors = plt.cm.get_cmap('tab10', NUM_CLASSES)
    
    for i, color in zip(range(NUM_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) for Multi-Class')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'roc_curves.png'))
    plt.close()
    print("ROC Curves saved.")


# --- Execute Training and Evaluation ---
print("Starting training...")
trained_model, history = train_model(model, criterion, optimizer, num_epochs=NUM_EPOCHS)

# --- Save Final Model and Generate Plots ---
print("Saving the best model...")
torch.save(trained_model.state_dict(), os.path.join(results_dir, 'arecanut_resnet50_best_weights.pth'))
print(f"Model saved to {os.path.join(results_dir, 'arecanut_resnet50_best_weights.pth')}")

print("\nGenerating and saving plots...")
plot_metrics(history, results_dir)
plot_confusion_matrix_and_roc(trained_model, dataloaders['val'], device, class_names, results_dir)

print("\nAll tasks completed. Check the 'ResNet_Results' directory.")