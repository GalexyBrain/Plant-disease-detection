# Plant-disease-detection


### 1. Directory layout for both experiments

I’ll name the model `DeeperCNN` (you can rename if you want).
We’ll treat **NonAugmented** = `dataset/` and **Augmented** = `augmented_dataset/`.

Here’s the **final structure** I suggest:

```text
NonAugmented/
└── DeeperCNN/
    ├── Model/
    │   ├── Deeper_CNN_Torch_Best.pt        # Best weights (non-augmented)
    │   └── model_config.json               # IMG size, LR, epochs, seed, etc.
    │
    ├── history_metrics.json                # Full training/val curves (per epoch)
    ├── summary_metrics.json                # Single-row summary of key metrics
    ├── per_class_metrics.csv               # Precision/Recall/F1/Support per class
    ├── auc_scores.csv                      # Per-class & macro/micro AUC scores
    │
    ├── training_curves_loss_acc.png        # Loss + Accuracy vs epoch (already have)
    ├── learning_rate_schedule.png          # LR vs epoch (from scheduler)
    ├── confusion_matrix.png                # Normalized confusion matrix (val set)
    ├── roc_curves_all_classes.png          # One-vs-rest ROC curves (already have)
    ├── pr_curves_all_classes.png           # One-vs-rest Precision–Recall curves
    ├── calibration_reliability_curve.png   # Reliability diagram (ECE style)
    ├── per_class_f1_barplot.png            # Bar chart of F1 per class
    └── misclassified_examples_topK/
        ├── top_confused_pairs.txt          # Which classes confuse with which
        └── *.png                           # Few grids of misclassified images (optional)

Augmented/
└── DeeperCNN/
    ├── Model/
    │   ├── Deeper_CNN_Torch_Best.pt        # Best weights (augmented data)
    │   └── model_config.json
    │
    ├── history_metrics.json
    ├── summary_metrics.json
    ├── per_class_metrics.csv
    ├── auc_scores.csv
    │
    ├── training_curves_loss_acc.png
    ├── learning_rate_schedule.png
    ├── confusion_matrix.png
    ├── roc_curves_all_classes.png
    ├── pr_curves_all_classes.png
    ├── calibration_reliability_curve.png
    ├── per_class_f1_barplot.png
    └── misclassified_examples_topK/
        ├── top_confused_pairs.txt
        └── *.png

Comparison/
└── DeeperCNN/
    ├── overall_metrics_comparison.csv      # One row per setup: NonAug vs Aug
    ├── per_class_metrics_comparison.csv    # F1/Recall per class for both, plus Δ
    ├── roc_auc_comparison.csv             # Per-class and macro/micro AUC for both
    │
    ├── overall_metrics_barplot.png         # Acc, Macro-F1, Kappa, MCC side-by-side
    ├── per_class_f1_comparison_barplot.png # Grouped bar: F1 per class (NonAug vs Aug)
    ├── per_class_auc_comparison_barplot.png# Grouped bar: AUC per class
    ├── delta_f1_barplot.png                # (Aug - NonAug) F1 per class
    ├── delta_auc_barplot.png               # (Aug - NonAug) AUC per class
    ├── calibration_comparison.png          # Two reliability curves on one plot
    └── confusion_matrix_side_by_side.png   # 2-panel figure: NonAug vs Aug confusion
```

You can put these three folders under a single `CNN_Results/` root if you want, but the core structure above is what the code will target.

---

### 2. Metrics to compute for **each** model

Using the validation predictions you already collect (`all_val_labels`, `all_val_probs`):

For **summary_metrics.json** (single-object JSON):

* `overall_accuracy`
* `balanced_accuracy`
* `macro_precision`, `macro_recall`, `macro_f1`
* `weighted_precision`, `weighted_recall`, `weighted_f1`
* `cohen_kappa`
* `matthews_corrcoef` (MCC)
* `macro_auc_ovr`, `weighted_auc_ovr` (from one-vs-rest ROC)
* `num_classes`, `class_names`
* `num_val_samples`
* `best_epoch`
* `best_val_acc`
* maybe `final_learning_rate`

For **per_class_metrics.csv**:

Columns:

* `class_name`
* `precision`
* `recall`
* `f1_score`
* `support` (number of samples)
* `auc_ovr` (AUC of that class one-vs-rest)
* `specificity` if you want to go extra (TN / (TN + FP) per class)

For **auc_scores.csv**:

* Same as you already do, but extended to also include macro/micro AUC in extra rows so it’s paper-friendly.

For **history_metrics.json**:

* Arrays indexed by epoch:

  * `train_loss`, `val_loss`
  * `train_acc`, `val_acc`
  * `learning_rate` (per epoch, from scheduler)

That’s enough to reproduce any figure later.

---

### 3. Graphs for **each** model (NonAugmented & Augmented)

These are the plots that scream “I belong in a Q1 paper”:

1. **training_curves_loss_acc.png**

   * Left: Train vs Val loss.
   * Right: Train vs Val accuracy.
   * All epochs actually run (respecting early stopping).
   * Proper titles, axis labels, legends, high DPI (300), consistent font size.

2. **learning_rate_schedule.png**

   * LR vs epoch; shows how ReduceLROnPlateau behaved.
   * Helps justify convergence behaviour.

3. **confusion_matrix.png**

   * Normalized (per true class).
   * Large fonts, rotated x-ticks, colorbar.
   * This is especially good for discussing which disease classes are hardest.

4. **roc_curves_all_classes.png** (already present)

   * One-vs-rest ROC curves for all 9 classes.
   * Legend with class name + AUC.
   * Diagonal no-skill dashed line.

5. **pr_curves_all_classes.png**

   * Precision–Recall curves one-vs-rest.
   * Very important if you ever mention class imbalance or rare diseases.

6. **calibration_reliability_curve.png**

   * Reliability diagram: predicted probability buckets vs empirical accuracy.
   * Optionally annotate Expected Calibration Error (ECE) in the title.

7. **per_class_f1_barplot.png**

   * Classes on x-axis, F1 on y-axis.
   * Shows which diseases are easiest/hardest, and sets up comparison with augmentation later.

8. **misclassified_examples_topK/** (optional but very nice)

   * For each of a few key confusions (e.g., `Mahali_Koleroga → Healthy_Leaf`),
     save small grids of misclassified images with predicted/true labels in the filename.
   * This supports qualitative analysis in the paper.

---

### 4. Comparison-level metrics & graphs

Using the two `summary_metrics.json`, `per_class_metrics.csv`, and `auc_scores.csv` files, the **Comparison/DeeperCNN/** folder will contain:

#### CSVs

* **overall_metrics_comparison.csv**

  Rows:

  * `setup` (NonAugmented vs Augmented)
  * `overall_accuracy`
  * `balanced_accuracy`
  * `macro_f1`
  * `weighted_f1`
  * `cohen_kappa`
  * `mcc`
  * `macro_auc_ovr`
  * `weighted_auc_ovr`

* **per_class_metrics_comparison.csv**

  Columns:

  * `class_name`
  * `precision_nonaug`, `recall_nonaug`, `f1_nonaug`, `auc_nonaug`
  * `precision_aug`, `recall_aug`, `f1_aug`, `auc_aug`
  * `delta_f1` (aug − nonaug)
  * `delta_auc` (aug − nonaug)

* **roc_auc_comparison.csv**

  * Per-class AUC values for both setups + macro/micro.

#### Plots

1. **overall_metrics_barplot.png**

   * Grouped bar chart: x-axis = metrics
     (Accuracy, Balanced Accuracy, Macro-F1, MCC, Macro-AUC).
   * Two bars per metric: NonAugmented vs Augmented.
   * This gives a clean, single-figure “augmentation helps” story.

2. **per_class_f1_comparison_barplot.png**

   * X-axis = classes.
   * Two bars per class: F1_nonaug vs F1_aug.
   * Shows which diseases benefit the most from augmentation.

3. **per_class_auc_comparison_barplot.png**

   * Similar grouped bar chart but for AUC per class.

4. **delta_f1_barplot.png**

   * Single bar per class showing `ΔF1 = F1_aug − F1_nonaug`.
   * Above zero = augmentation helped that class, below zero = hurt.
   * This is *very* Q1-friendly for “per-class impact of data augmentation.”

5. **delta_auc_barplot.png**

   * Same idea but for AUC.

6. **calibration_comparison.png**

   * Overlay two reliability curves (NonAug vs Aug) in a single plot.
   * Shows whether augmentation changed calibration quality.

7. **confusion_matrix_side_by_side.png**

   * Two subplots (left: NonAug, right: Aug).
   * Normalized CMs with same color scale.
   * Nice visual for “augmentation reduces confusion between X and Y”.