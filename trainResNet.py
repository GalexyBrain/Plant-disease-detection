import os
import copy
import csv
import json
from datetime import datetime
from typing import Dict, List
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from tqdm import tqdm

# ----------------- GLOBAL CONFIG -----------------
NUM_CLASSES = 9  # matches your dataset (9 classes)
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-5
MODEL_NAME = "ResNet50"

# Number of DataLoader workers
NUM_WORKERS = max(0, max(4, (os.cpu_count() or 4)))

sns.set_theme(style="whitegrid")


# ----------------- DATA & MODEL HELPERS -----------------
def build_dataloaders(data_root: str):
    """
    data_root/
      train/
      val/
    """
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        ),
    }

    image_datasets = {
        split: datasets.ImageFolder(
            os.path.join(data_root, split), data_transforms[split]
        )
        for split in ["train", "val"]
    }

    pin_memory = torch.cuda.is_available()
    persistent_workers = NUM_WORKERS > 0

    dataloaders = {
        split: DataLoader(
            image_datasets[split],
            batch_size=BATCH_SIZE,
            shuffle=True if split == "train" else False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        for split in ["train", "val"]
    }

    dataset_sizes = {split: len(image_datasets[split]) for split in ["train", "val"]}
    class_names = image_datasets["train"].classes

    return dataloaders, dataset_sizes, class_names


def init_model(num_classes: int, device: torch.device):
    """
    ResNet50 trained from scratch (weights=None), final FC adapted to num_classes.
    """
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    return model


# ----------------- TRAIN A SINGLE EXPERIMENT -----------------
def train_one_experiment(
    exp_root: str,
    data_root: str,
    device: torch.device,
):
    """
    Trains ResNet50 on one dataset root (dataset or augmented_dataset),
    saves:
      - Model/ResNet50_Best.pth
      - Model/model_config.json
      - history_metrics.json
      - summary_metrics.json
      - per_class_metrics.csv
      - auc_scores.csv
      - plots + confusion matrix + calibration, etc.
    """
    os.makedirs(exp_root, exist_ok=True)
    model_dir = os.path.join(exp_root, "Model")
    os.makedirs(model_dir, exist_ok=True)

    dataloaders, dataset_sizes, class_names = build_dataloaders(data_root)
    num_classes = len(class_names)

    print(f"\n📂 Training on data root: {data_root}")
    print(f"📁 Experiment directory: {exp_root}")
    print(f"🔢 Detected classes ({num_classes}): {class_names}")

    model = init_model(num_classes, device)
    print("✅ Model initialized:\n", model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.2, patience=4
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    lr_history: List[float] = []

    best_acc = 0.0
    best_epoch = -1
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    early_stopping_patience = 50  # same as your previous setting

    all_val_labels_best = None
    all_val_probs_best = None

    for epoch in range(NUM_EPOCHS):
        print(f"\n🔥 Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.8f}")

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            epoch_labels_list = []
            epoch_probs_list = []

            progress_bar = tqdm(
                dataloaders[phase],
                desc=f"{phase.capitalize()} Epoch {epoch+1}/{NUM_EPOCHS}",
            )

            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if phase == "val":
                    probs = torch.softmax(outputs, 1).detach().cpu()
                    epoch_probs_list.append(probs)
                    epoch_labels_list.append(labels.detach().cpu())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double().item() / dataset_sizes[phase]

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val":
                # Step LR scheduler on validation accuracy
                scheduler.step(epoch_acc)
                lr_history.append(optimizer.param_groups[0]["lr"])

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch + 1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(
                        f"✅ New best model with validation accuracy: {best_acc:.4f} at epoch {best_epoch}"
                    )
                    # save best epoch predictions
                    all_val_labels_best = torch.cat(epoch_labels_list).numpy()
                    all_val_probs_best = torch.cat(epoch_probs_list).numpy()
                    # save best weights
                    torch.save(
                        best_model_wts,
                        os.path.join(model_dir, f"{MODEL_NAME}_Best.pth"),
                    )
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        # Early stopping (with big patience)
        if epochs_no_improve >= early_stopping_patience:
            print(f"\n✋ Early stopping at epoch {epoch+1}")
            break

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n🎉 Best val Acc: {best_acc:.4f} at epoch {best_epoch}")
    model.load_state_dict(best_model_wts)

    # Save model config
    model_config = {
        "model_name": MODEL_NAME,
        "num_classes": num_classes,
        "class_names": class_names,
        "batch_size": BATCH_SIZE,
        "num_epochs_planned": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "dataset_root": data_root,
        "device": str(device),
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
    }
    with open(os.path.join(model_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    # Save full history
    history_metrics = {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "train_acc": history["train_acc"],
        "val_acc": history["val_acc"],
        "lr": lr_history,
    }
    with open(os.path.join(exp_root, "history_metrics.json"), "w") as f:
        json.dump(history_metrics, f, indent=4)

    # Save validation predictions for best epoch and generate all Q1-level plots/metrics
    if all_val_labels_best is not None and all_val_probs_best is not None:
        np.savez(
            os.path.join(exp_root, "val_predictions_best_epoch.npz"),
            labels=all_val_labels_best,
            probs=all_val_probs_best,
        )

        generate_experiment_reports(
            exp_root=exp_root,
            class_names=class_names,
            all_val_labels=all_val_labels_best,
            all_val_probs=all_val_probs_best,
            history=history,
            lr_history=lr_history,
        )
    else:
        print("⚠️ No validation predictions collected for best epoch.")

    return {
        "exp_root": exp_root,
        "data_root": data_root,
        "class_names": class_names,
        "num_classes": num_classes,
    }


# ----------------- EXPERIMENT-LEVEL REPORTS -----------------
def generate_experiment_reports(
    exp_root: str,
    class_names: List[str],
    all_val_labels: np.ndarray,
    all_val_probs: np.ndarray,
    history: Dict[str, List[float]],
    lr_history: List[float],
):
    """
    Generate all metrics & plots for a single experiment:
      - summary_metrics.json
      - per_class_metrics.csv
      - auc_scores.csv
      - confusion_matrix (png + npy)
      - training_curves_loss_acc.png
      - learning_rate_schedule.png
      - roc_curves_all_classes.png
      - pr_curves_all_classes.png
      - per_class_f1_barplot.png
      - calibration_curve.json + calibration_reliability_curve.png
      - classification_report.txt
      - misclassified_examples_topK/top_confused_pairs.txt
    """
    num_classes = len(class_names)
    os.makedirs(exp_root, exist_ok=True)

    all_val_labels = np.asarray(all_val_labels)
    all_val_probs = np.asarray(all_val_probs)
    all_val_preds = np.argmax(all_val_probs, axis=1)

    # ----- summary metrics -----
    overall_accuracy = accuracy_score(all_val_labels, all_val_preds)
    balanced_acc = balanced_accuracy_score(all_val_labels, all_val_preds)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_val_labels, all_val_preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            all_val_labels, all_val_preds, average="weighted", zero_division=0
        )
    )

    kappa = cohen_kappa_score(all_val_labels, all_val_preds)
    mcc = matthews_corrcoef(all_val_labels, all_val_preds)

    # One-vs-rest ROC, AUC and AP
    y_true_bin = label_binarize(all_val_labels, classes=list(range(num_classes)))
    fpr = {}
    tpr = {}
    roc_auc_vals = {}
    ap_scores = {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_val_probs[:, i])
        roc_auc_vals[i] = auc(fpr[i], tpr[i])
        ap_scores[i] = average_precision_score(y_true_bin[:, i], all_val_probs[:, i])

    per_class_aucs = np.array([roc_auc_vals[i] for i in range(num_classes)])
    macro_auc = float(np.mean(per_class_aucs))
    class_counts = np.bincount(all_val_labels, minlength=num_classes)
    weights = class_counts / class_counts.sum()
    weighted_auc = float(np.sum(per_class_aucs * weights))

    report_dict = classification_report(
        all_val_labels,
        all_val_preds,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    summary_metrics = {
        "overall_accuracy": overall_accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "weighted_precision": precision_weighted,
        "weighted_recall": recall_weighted,
        "weighted_f1": f1_weighted,
        "cohen_kappa": kappa,
        "matthews_corrcoef": mcc,
        "macro_auc_ovr": macro_auc,
        "weighted_auc_ovr": weighted_auc,
        "num_classes": num_classes,
        "num_val_samples": int(len(all_val_labels)),
    }
    with open(os.path.join(exp_root, "summary_metrics.json"), "w") as f:
        json.dump(summary_metrics, f, indent=4)

    # ----- per-class metrics CSV -----
    per_class_csv_path = os.path.join(exp_root, "per_class_metrics.csv")
    with open(per_class_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "class_index",
                "class_name",
                "precision",
                "recall",
                "f1_score",
                "support",
                "auc_ovr",
                "average_precision",
            ]
        )
        for idx, cname in enumerate(class_names):
            metrics = report_dict.get(cname, {})
            writer.writerow(
                [
                    idx,
                    cname,
                    metrics.get("precision", 0.0),
                    metrics.get("recall", 0.0),
                    metrics.get("f1-score", 0.0),
                    metrics.get("support", 0),
                    roc_auc_vals[idx],
                    ap_scores[idx],
                ]
            )

    # ----- AUC scores CSV -----
    auc_csv_path = os.path.join(exp_root, "auc_scores.csv")
    with open(auc_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Class", "AUC"])
        for idx, cname in enumerate(class_names):
            writer.writerow([cname, f"{roc_auc_vals[idx]:.4f}"])
        writer.writerow(["macro_ovr", f"{macro_auc:.4f}"])
        writer.writerow(["weighted_ovr", f"{weighted_auc:.4f}"])

    # ----- Confusion matrix (raw + normalized + plot) -----
    cm = confusion_matrix(all_val_labels, all_val_preds)
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
    np.save(os.path.join(exp_root, "confusion_matrix_raw.npy"), cm)
    np.save(os.path.join(exp_root, "confusion_matrix_norm.npy"), cm_norm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title("Normalized Confusion Matrix", fontsize=16)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(exp_root, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # ----- Training curves: loss & accuracy -----
    num_epochs = len(history["train_loss"])
    epochs_range = range(1, num_epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(epochs_range, history["train_loss"], "o-", label="Train Loss")
    ax1.plot(epochs_range, history["val_loss"], "o-", label="Val Loss")
    ax1.set_title("Model Loss", fontsize=16)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True)

    ax2.plot(epochs_range, history["train_acc"], "o-", label="Train Acc")
    ax2.plot(epochs_range, history["val_acc"], "o-", label="Val Acc")
    ax2.set_title("Model Accuracy", fontsize=16)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True)

    fig.suptitle("Training and Validation Metrics", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    metrics_plot_path = os.path.join(exp_root, "training_curves_loss_acc.png")
    plt.savefig(metrics_plot_path, dpi=300)
    plt.close()

    # ----- Learning rate schedule -----
    if lr_history:
        lr_epochs = range(1, len(lr_history) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(lr_epochs, lr_history, "o-")
        plt.title("Learning Rate Schedule", fontsize=16)
        plt.xlabel("Epoch (val steps)", fontsize=12)
        plt.ylabel("Learning Rate", fontsize=12)
        plt.grid(True)
        lr_plot_path = os.path.join(exp_root, "learning_rate_schedule.png")
        plt.savefig(lr_plot_path, dpi=300)
        plt.close()

    # ----- ROC curves (OvR) -----
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", num_classes)
    for i, (cname, color) in enumerate(zip(class_names, colors)):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"{cname} (AUC = {roc_auc_vals[i]:.3f})",
        )
    plt.plot([0, 1], [0, 1], "k--", lw=2, label="No-Skill")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Multi-Class ROC Curves (OvR)", fontsize=16)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True)
    roc_plot_path = os.path.join(exp_root, "roc_curves_all_classes.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()

    # ----- Precision–Recall curves (OvR) -----
    plt.figure(figsize=(10, 8))
    for i, (cname, color) in enumerate(zip(class_names, colors)):
        precision_i, recall_i, _ = precision_recall_curve(
            y_true_bin[:, i], all_val_probs[:, i]
        )
        plt.plot(
            recall_i,
            precision_i,
            color=color,
            lw=2,
            label=f"{cname} (AP = {ap_scores[i]:.3f})",
        )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Multi-Class Precision–Recall Curves (OvR)", fontsize=16)
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True)
    pr_plot_path = os.path.join(exp_root, "pr_curves_all_classes.png")
    plt.savefig(pr_plot_path, dpi=300)
    plt.close()

    # ----- Per-class F1 barplot -----
    f1_scores = [report_dict[c]["f1-score"] for c in class_names]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=f1_scores)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("F1 Score", fontsize=12)
    plt.title("Per-Class F1 Scores", fontsize=16)
    plt.tight_layout()
    f1_barplot_path = os.path.join(exp_root, "per_class_f1_barplot.png")
    plt.savefig(f1_barplot_path, dpi=300)
    plt.close()

    # ----- Calibration / reliability diagram -----
    max_probs = np.max(all_val_probs, axis=1)
    correctness = (all_val_preds == all_val_labels).astype(int)
    prob_true, prob_pred = calibration_curve(
        correctness, max_probs, n_bins=10, strategy="uniform"
    )
    calib_data = {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }
    with open(os.path.join(exp_root, "calibration_curve.json"), "w") as f:
        json.dump(calib_data, f, indent=4)

    plt.figure(figsize=(7, 7))
    plt.plot(prob_pred, prob_true, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Empirical Accuracy", fontsize=12)
    plt.title("Reliability Diagram (Top-1 Confidence)", fontsize=16)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True)
    calib_plot_path = os.path.join(exp_root, "calibration_reliability_curve.png")
    plt.savefig(calib_plot_path, dpi=300)
    plt.close()

    # ----- Classification report (text) -----
    report_str = classification_report(
        all_val_labels,
        all_val_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_path = os.path.join(exp_root, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Classification Report\n")
        f.write("=====================\n\n")
        f.write(report_str)

    # ----- Misclassified top-confused pairs (text only) -----
    mis_dir = os.path.join(exp_root, "misclassified_examples_topK")
    os.makedirs(mis_dir, exist_ok=True)
    cm_counts = cm.copy()
    np.fill_diagonal(cm_counts, 0)
    pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm_counts[i, j] > 0:
                pairs.append((cm_counts[i, j], class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    top_path = os.path.join(mis_dir, "top_confused_pairs.txt")
    with open(top_path, "w") as f:
        f.write("Top confused class pairs (count, true, predicted):\n")
        for count, true_c, pred_c in pairs[:20]:
            f.write(f"{count:5d}  {true_c} -> {pred_c}\n")


# ----------------- LOAD HELPERS FOR COMPARISON -----------------
def load_summary_metrics(exp_root: str) -> Dict:
    with open(os.path.join(exp_root, "summary_metrics.json"), "r") as f:
        return json.load(f)


def load_per_class_metrics(exp_root: str) -> List[Dict]:
    rows = []
    with open(os.path.join(exp_root, "per_class_metrics.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_auc_scores(exp_root: str) -> Dict[str, float]:
    scores = {}
    with open(os.path.join(exp_root, "auc_scores.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores[row["Class"]] = float(row["AUC"])
    return scores


def load_calibration_data(exp_root: str) -> Dict[str, List[float]]:
    with open(os.path.join(exp_root, "calibration_curve.json"), "r") as f:
        return json.load(f)


# ----------------- COMPARISON BETWEEN NON-AUG & AUG -----------------
def compare_experiments(
    nonaug_root: str,
    aug_root: str,
    comparison_root: str,
    class_names: List[str],
):
    """
    Build Comparison/ResNet50/ with:
      - overall_metrics_comparison.csv
      - per_class_metrics_comparison.csv
      - overall_metrics_barplot.png
      - per_class_f1_comparison_barplot.png
      - per_class_auc_comparison_barplot.png
      - delta_f1_barplot.png
      - delta_auc_barplot.png
      - calibration_comparison.png
      - confusion_matrix_side_by_side.png
    """
    os.makedirs(comparison_root, exist_ok=True)

    metrics_nonaug = load_summary_metrics(nonaug_root)
    metrics_aug = load_summary_metrics(aug_root)

    per_class_nonaug = load_per_class_metrics(nonaug_root)
    per_class_aug = load_per_class_metrics(aug_root)

    auc_nonaug = load_auc_scores(nonaug_root)
    auc_aug = load_auc_scores(aug_root)

    calib_nonaug = load_calibration_data(nonaug_root)
    calib_aug = load_calibration_data(aug_root)

    # ----- overall metrics comparison CSV -----
    overall_csv_path = os.path.join(comparison_root, "overall_metrics_comparison.csv")
    overall_metrics = [
        "overall_accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "cohen_kappa",
        "matthews_corrcoef",
        "macro_auc_ovr",
        "weighted_auc_ovr",
    ]
    key_map = {
        "overall_accuracy": "overall_accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "macro_f1",
        "weighted_f1": "weighted_f1",
        "cohen_kappa": "cohen_kappa",
        "matthews_corrcoef": "matthews_corrcoef",
        "macro_auc_ovr": "macro_auc_ovr",
        "weighted_auc_ovr": "weighted_auc_ovr",
    }
    with open(overall_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["setup"] + overall_metrics)
        row_nonaug = ["NonAugmented"]
        row_aug = ["Augmented"]
        for m in overall_metrics:
            key = key_map[m]
            row_nonaug.append(metrics_nonaug.get(key, 0.0))
            row_aug.append(metrics_aug.get(key, 0.0))
        writer.writerow(row_nonaug)
        writer.writerow(row_aug)

    # ----- per-class comparison CSV -----
    per_class_cmp_csv = os.path.join(
        comparison_root, "per_class_metrics_comparison.csv"
    )
    with open(per_class_cmp_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "class_index",
                "class_name",
                "precision_nonaug",
                "recall_nonaug",
                "f1_nonaug",
                "auc_nonaug",
                "precision_aug",
                "recall_aug",
                "f1_aug",
                "auc_aug",
                "delta_f1",
                "delta_auc",
            ]
        )

        nonaug_by_name = {row["class_name"]: row for row in per_class_nonaug}
        aug_by_name = {row["class_name"]: row for row in per_class_aug}

        for idx, cname in enumerate(class_names):
            row_na = nonaug_by_name.get(cname, {})
            row_a = aug_by_name.get(cname, {})

            f1_na = float(row_na.get("f1_score", 0.0)) if row_na else 0.0
            f1_a = float(row_a.get("f1_score", 0.0)) if row_a else 0.0
            auc_na = auc_nonaug.get(cname, 0.0)
            auc_a = auc_aug.get(cname, 0.0)

            writer.writerow(
                [
                    idx,
                    cname,
                    float(row_na.get("precision", 0.0)) if row_na else 0.0,
                    float(row_na.get("recall", 0.0)) if row_na else 0.0,
                    f1_na,
                    auc_na,
                    float(row_a.get("precision", 0.0)) if row_a else 0.0,
                    float(row_a.get("recall", 0.0)) if row_a else 0.0,
                    f1_a,
                    auc_a,
                    f1_a - f1_na,
                    auc_a - auc_na,
                ]
            )

    # ----- grouped overall metrics barplot -----
    metric_labels = [
        "Accuracy",
        "Balanced Accuracy",
        "Macro F1",
        "Weighted F1",
        "Kappa",
        "MCC",
        "Macro AUC",
        "Weighted AUC",
    ]
    nonaug_vals = [
        metrics_nonaug.get("overall_accuracy", 0.0),
        metrics_nonaug.get("balanced_accuracy", 0.0),
        metrics_nonaug.get("macro_f1", 0.0),
        metrics_nonaug.get("weighted_f1", 0.0),
        metrics_nonaug.get("cohen_kappa", 0.0),
        metrics_nonaug.get("matthews_corrcoef", 0.0),
        metrics_nonaug.get("macro_auc_ovr", 0.0),
        metrics_nonaug.get("weighted_auc_ovr", 0.0),
    ]
    aug_vals = [
        metrics_aug.get("overall_accuracy", 0.0),
        metrics_aug.get("balanced_accuracy", 0.0),
        metrics_aug.get("macro_f1", 0.0),
        metrics_aug.get("weighted_f1", 0.0),
        metrics_aug.get("cohen_kappa", 0.0),
        metrics_aug.get("matthews_corrcoef", 0.0),
        metrics_aug.get("macro_auc_ovr", 0.0),
        metrics_aug.get("weighted_auc_ovr", 0.0),
    ]
    x = np.arange(len(metric_labels))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, nonaug_vals, width, label="NonAugmented")
    plt.bar(x + width / 2, aug_vals, width, label="Augmented")
    plt.xticks(x, metric_labels, rotation=45, ha="right")
    plt.ylabel("Score", fontsize=12)
    plt.title("Overall Metrics Comparison (NonAugmented vs Augmented)", fontsize=16)
    plt.legend()
    plt.tight_layout()
    overall_barplot_path = os.path.join(
        comparison_root, "overall_metrics_barplot.png"
    )
    plt.savefig(overall_barplot_path, dpi=300)
    plt.close()

    # ----- per-class F1 comparison plot -----
    nonaug_by_name = {row["class_name"]: row for row in per_class_nonaug}
    aug_by_name = {row["class_name"]: row for row in per_class_aug}
    f1_nonaug = []
    f1_aug_vals = []
    for cname in class_names:
        row_na = nonaug_by_name.get(cname, {})
        row_a = aug_by_name.get(cname, {})
        f1_nonaug.append(float(row_na.get("f1_score", 0.0)) if row_na else 0.0)
        f1_aug_vals.append(float(row_a.get("f1_score", 0.0)) if row_a else 0.0)

    x = np.arange(len(class_names))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, f1_nonaug, width, label="NonAugmented")
    plt.bar(x + width / 2, f1_aug_vals, width, label="Augmented")
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylabel("F1 Score", fontsize=12)
    plt.title("Per-Class F1 Comparison", fontsize=16)
    plt.legend()
    plt.tight_layout()
    f1_cmp_path = os.path.join(
        comparison_root, "per_class_f1_comparison_barplot.png"
    )
    plt.savefig(f1_cmp_path, dpi=300)
    plt.close()

    # ----- per-class AUC comparison plot -----
    auc_nonaug_vals = [auc_nonaug.get(c, 0.0) for c in class_names]
    auc_aug_vals = [auc_aug.get(c, 0.0) for c in class_names]
    x = np.arange(len(class_names))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, auc_nonaug_vals, width, label="NonAugmented")
    plt.bar(x + width / 2, auc_aug_vals, width, label="Augmented")
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylabel("AUC (OvR)", fontsize=12)
    plt.title("Per-Class AUC Comparison", fontsize=16)
    plt.legend()
    plt.tight_layout()
    auc_cmp_path = os.path.join(
        comparison_root, "per_class_auc_comparison_barplot.png"
    )
    plt.savefig(auc_cmp_path, dpi=300)
    plt.close()

    # ----- ΔF1 barplot -----
    delta_f1 = [a - n for a, n in zip(f1_aug_vals, f1_nonaug)]
    plt.figure(figsize=(12, 6))
    sns.barplot(x=class_names, y=delta_f1)
    plt.xticks(rotation=45, ha="right")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.ylabel("ΔF1 (Aug - NonAug)", fontsize=12)
    plt.title("Per-Class ΔF1 (Impact of Augmentation)", fontsize=16)
    plt.tight_layout()
    delta_f1_path = os.path.join(comparison_root, "delta_f1_barplot.png")
    plt.savefig(delta_f1_path, dpi=300)
    plt.close()

    # ----- ΔAUC barplot -----
    delta_auc = [a - n for a, n in zip(auc_aug_vals, auc_nonaug_vals)]
    plt.figure(figsize=(12, 6))
    sns.barplot(x=class_names, y=delta_auc)
    plt.xticks(rotation=45, ha="right")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.ylabel("ΔAUC (Aug - NonAug)", fontsize=12)
    plt.title("Per-Class ΔAUC (Impact of Augmentation)", fontsize=16)
    plt.tight_layout()
    delta_auc_path = os.path.join(comparison_root, "delta_auc_barplot.png")
    plt.savefig(delta_auc_path, dpi=300)
    plt.close()

    # ----- Calibration comparison -----
    prob_true_na = np.array(calib_nonaug["prob_true"])
    prob_pred_na = np.array(calib_nonaug["prob_pred"])
    prob_true_a = np.array(calib_aug["prob_true"])
    prob_pred_a = np.array(calib_aug["prob_pred"])

    plt.figure(figsize=(7, 7))
    plt.plot(prob_pred_na, prob_true_na, "s-", label="NonAugmented")
    plt.plot(prob_pred_a, prob_true_a, "o-", label="Augmented")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Empirical Accuracy", fontsize=12)
    plt.title("Calibration Comparison (Reliability Diagram)", fontsize=16)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True)
    calib_cmp_path = os.path.join(comparison_root, "calibration_comparison.png")
    plt.savefig(calib_cmp_path, dpi=300)
    plt.close()

    # ----- Confusion matrices side-by-side -----
    cm_nonaug = np.load(os.path.join(nonaug_root, "confusion_matrix_norm.npy"))
    cm_aug = np.load(os.path.join(aug_root, "confusion_matrix_norm.npy"))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

    sns.heatmap(
        cm_nonaug,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=axes[0],
    )
    axes[0].set_title("NonAugmented", fontsize=14)
    axes[0].set_ylabel("True Label", fontsize=12)
    axes[0].set_xlabel("Predicted Label", fontsize=12)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(
        cm_aug,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=axes[1],
    )
    axes[1].set_title("Augmented", fontsize=14)
    axes[1].set_ylabel("True Label", fontsize=12)
    axes[1].set_xlabel("Predicted Label", fontsize=12)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    fig.suptitle("Normalized Confusion Matrices", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    cm_cmp_path = os.path.join(
        comparison_root, "confusion_matrix_side_by_side.png"
    )
    plt.savefig(cm_cmp_path, dpi=300)
    plt.close()


# ----------------- MAIN -----------------
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    print(f"PyTorch Version: {torch.__version__}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root_results_dir = f"."
    os.makedirs(root_results_dir, exist_ok=True)
    print(f"📂 All results will be saved in: {root_results_dir}")

    # Non-augmented experiment: dataset/
    nonaug_data_root = "dataset"
    nonaug_root = os.path.join(root_results_dir, "NonAugmented", MODEL_NAME)
    nonaug_info = train_one_experiment(
        exp_root=nonaug_root,
        data_root=nonaug_data_root,
        device=device,
    )

    # Augmented experiment: augmented_dataset/
    aug_data_root = "augmented_dataset"
    aug_root = os.path.join(root_results_dir, "Augmented", MODEL_NAME)
    aug_info = train_one_experiment(
        exp_root=aug_root,
        data_root=aug_data_root,
        device=device,
    )

    # Ensure class order is consistent
    class_names_nonaug = nonaug_info["class_names"]
    class_names_aug = aug_info["class_names"]
    if class_names_nonaug != class_names_aug:
        print("⚠️ WARNING: Class names differ between experiments.")
        print("NonAugmented classes:", class_names_nonaug)
        print("Augmented classes:", class_names_aug)
    class_names = class_names_nonaug

    # Comparison folder
    comparison_root = os.path.join(root_results_dir, "Comparison", MODEL_NAME)
    compare_experiments(
        nonaug_root=nonaug_root,
        aug_root=aug_root,
        comparison_root=comparison_root,
        class_names=class_names,
    )

    print("\n✅ All tasks completed.")
    print("Root results directory:", root_results_dir)


if __name__ == "__main__":
    main()
