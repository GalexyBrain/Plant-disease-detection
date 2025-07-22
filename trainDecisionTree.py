import os
import torch
from torchvision import datasets, models, transforms
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 📁 Paths
train_dir = "./augmented_dataset/train"
val_dir = "./augmented_dataset/val"
output_dir = "./DecisionTreeOutput"
os.makedirs(output_dir, exist_ok=True)

# 📦 Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 📂 Load Data
train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

# 🧠 Feature extractor: ResNet18 (no FC layer)
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

# 📊 Extract features
print("🔍 Extracting features...")
X_train, y_train = extract_features(train_loader)
X_val, y_val = extract_features(val_loader)

# 🌳 Train Decision Tree
clf = DecisionTreeClassifier(random_state=42, max_depth=20)
clf.fit(X_train, y_train)

# 🎯 Predict
y_pred = clf.predict(X_val)

# 📈 Accuracy
acc_train = clf.score(X_train, y_train)
acc_val = accuracy_score(y_val, y_pred)

# 📝 Save classification report
report = classification_report(y_val, y_pred, target_names=val_data.classes)
with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# 🔍 Save confusion matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=val_data.classes, yticklabels=val_data.classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# 📊 Save accuracy bar chart
plt.figure()
plt.bar(["Train Accuracy", "Validation Accuracy"], [acc_train, acc_val], color=["green", "orange"])
plt.title("Decision Tree Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
plt.close()

print(f"\n✅ All outputs saved to {output_dir}/")

