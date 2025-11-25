import os
import torch
from torchvision import datasets, models, transforms
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 📁 Directories
train_dir = "./augmented_dataset/train"
val_dir = "./augmented_dataset/val"
output_dir = "./SVMOutput"
os.makedirs(output_dir, exist_ok=True)

# 🌈 Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 📂 Load dataset
train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

# 🧠 ResNet18 as feature extractor
resnet = models.resnet18()
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

def extract_features(dataloader):
    features, labels = [], []
    with torch.no_grad():
        for images, targets in dataloader:
            outputs = resnet(images)
            outputs = outputs.view(images.size(0), -1)
            features.append(outputs.numpy())
            labels.extend(targets.numpy())
    return np.vstack(features), np.array(labels)

# 🔍 Feature extraction
print("🔍 Extracting features...")
X_train, y_train = extract_features(train_loader)
X_val, y_val = extract_features(val_loader)

# ⚙️ Preprocessing: Scale + PCA (optional)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

# 🚀 Train SVM
print("🤖 Training SVM...")
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train_pca, y_train)

# 🎯 Predictions
y_pred = svm.predict(X_val_pca)

# 📈 Accuracy
acc_train = svm.score(X_train_pca, y_train)
acc_val = accuracy_score(y_val, y_pred)

# 📋 Save classification report
report = classification_report(y_val, y_pred, target_names=val_data.classes)
with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# 🧾 Save confusion matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
            xticklabels=val_data.classes, yticklabels=val_data.classes)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# 📊 Save accuracy plot
plt.figure()
plt.bar(["Train Accuracy", "Validation Accuracy"], [acc_train, acc_val], color=["blue", "red"])
plt.title("SVM Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
plt.close()

print(f"\n✅ All SVM results saved in {output_dir}/")

