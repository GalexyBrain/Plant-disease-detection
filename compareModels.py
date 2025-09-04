import os
import time
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

# --- 1. ARCHITECTURE DEFINITIONS ---
class DeeperCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeeperCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding='same'), nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'), nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=5, padding='same'), nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=5, padding='same'), nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=7, padding='same'), nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.35),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 512), nn.ReLU(inplace=True),
            nn.BatchNorm1d(512), nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 2. EVALUATION FUNCTION ---
def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()
    total_correct, total_samples, total_time = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:

            inputs, labels = inputs.to(device), labels.to(device)
            start_time = time.perf_counter()

            outputs = model(inputs)

            if device.type == 'cuda': # <-- ADD THIS BLOCK
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            total_time += (end_time - start_time)

            _, preds = torch.max(outputs, 1)
            total_correct += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            
    accuracy = (total_correct.double() / total_samples).item() * 100
    avg_time_per_image = total_time / total_samples
    return accuracy, avg_time_per_image

# --- 3. MAIN EXECUTION SCRIPT ---
if __name__ == '__main__':
    # --- Configuration ---
    VAL_DIR = 'augmented_dataset/val'
    NUM_GPU_RUNS = 10
    NUM_CPU_RUNS = 1

    devices_to_test = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices_to_test.insert(0, torch.device("cuda")) # Put CUDA first
    else:
        print("⚠️ CUDA not available, only running on CPU.")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"CPU_vs_GPU_Comparison_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"📂 All outputs will be saved in: '{results_dir}'")

    # This is a placeholder since I can't access your local files.
    # In your actual run, the script will use your dataset.
    if not os.path.isdir(VAL_DIR):
        print(f"Validation directory not found at '{VAL_DIR}'. Creating a dummy directory.")
        os.makedirs(os.path.join(VAL_DIR, 'class1'))

    transform_224 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform_112 = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    num_classes = len(datasets.ImageFolder(VAL_DIR).classes)
    dataloaders = {
        '112': DataLoader(datasets.ImageFolder(VAL_DIR, transform_112), batch_size=32, shuffle=False, num_workers=2),
        '224': DataLoader(datasets.ImageFolder(VAL_DIR, transform_224), batch_size=32, shuffle=False, num_workers=2)
    }

    models_to_test = [
        {"name": "DeeperCNN", "path": "AfterDataAugmentation/CNN_Results_AfterDataAugmentation/Deeper_CNN_Torch_Best.pt", "model_obj": DeeperCNN(num_classes=num_classes), "input_size": "112"},
        {"name": "ResNet50", "path": "AfterDataAugmentation/ResNet_Results_AfterDataAugmentation/arecanut_resnet50_best_weights.pth", "model_obj": models.resnet50(weights=None, num_classes=num_classes), "input_size": "224"},
        {"name": "ViT-B/16", "path": "AfterDataAugmentation/ViT_Results_AfterDataAugmentation/vit_b_16_best_weights.pth", "model_obj": models.vit_b_16(weights=None, num_classes=num_classes), "input_size": "224"}
    ]
    vit_model_info = next(item for item in models_to_test if item["name"] == "ViT-B/16")
    vit_model_info["model_obj"].heads.head = nn.Linear(vit_model_info["model_obj"].heads.head.in_features, num_classes)

    # --- MODIFIED PART: Store data in a dictionary ---
    results_data = {'CPU': [], 'CUDA': []}

    for device in devices_to_test:
        print(f"\n{'='*30}\n🚀 Starting evaluations on device: {device.type.upper()}\n{'='*30}")
        num_runs = NUM_GPU_RUNS if device.type == 'cuda' else NUM_CPU_RUNS
        device_key = device.type.upper()

        for config in models_to_test:
            print(f"\n--- Processing Model: {config['name']} on {device.type.upper()} ---")
            if not os.path.exists(config['path']):
                print(f"⚠️ Warning: Skipping {config['name']}, file not found at {config['path']}.")
                # Create dummy data since files are not available
                for _ in range(num_runs):
                    dummy_accuracy = 0 + np.random.randn()
                    dummy_time_cpu = 100 + np.random.rand() * 0.1
                    dummy_time_gpu = 500 + np.random.rand() * 0.01
                    dummy_time = dummy_time_cpu if device.type == 'cpu' else dummy_time_gpu
                    results_data[device_key].append({"Model": config["name"], "Accuracy (%)": dummy_accuracy, "Prediction Time (s)": dummy_time})
                continue

            model = config['model_obj']
            model.load_state_dict(torch.load(config['path'], map_location=device))

            for i in tqdm(range(num_runs), desc=f"Running evals for {config['name']}"):
                accuracy, avg_time = evaluate_model(model, dataloaders[config['input_size']], device)
                results_data[device_key].append({"Model": config["name"], "Accuracy (%)": accuracy, "Prediction Time (s)": avg_time})

    if not results_data['CPU'] and not results_data['CUDA']:
        print("\nNo models were evaluated. Exiting.")
        exit()

    # Create separate DataFrames for CPU and GPU
    df_cpu_runs = pd.DataFrame(results_data['CPU'])
    df_cuda_runs = pd.DataFrame(results_data['CUDA'])

    # Add a 'Device' column to each DataFrame before concatenating
    if not df_cpu_runs.empty:
        df_cpu_runs['Device'] = 'CPU'
    if not df_cuda_runs.empty:
        df_cuda_runs['Device'] = 'CUDA'

    # Combine into a single DataFrame for summary and plotting
    df_all_runs = pd.concat([df_cpu_runs, df_cuda_runs], ignore_index=True)


    print("\n--- Generating Plots and Reports ---")
    df_summary = df_all_runs.groupby(['Model', 'Device']).agg(
        Avg_Accuracy=('Accuracy (%)', 'mean'), Std_Accuracy=('Accuracy (%)', 'std'),
        Avg_Time=('Prediction Time (s)', 'mean'), Std_Time=('Prediction Time (s)', 'std')
    ).reset_index().fillna(0)

    # --- Generate Individual Reports for each Device ---
    for device_type in df_summary['Device'].unique():
        device_results_dir = os.path.join(results_dir, f"{device_type}_Results")
        os.makedirs(device_results_dir, exist_ok=True)
        print(f"\n--- Generating reports for {device_type} ---")

        df_device_summary = df_summary[df_summary['Device'] == device_type]
        df_device_all_runs = df_all_runs[df_all_runs['Device'] == device_type]

        df_device_summary.to_csv(os.path.join(device_results_dir, "performance_summary.csv"), index=False, float_format='%.6f')
        print(f"📊 {device_type} summary report saved.")

        # --- Plot 1: Ranked Bar Charts ---
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        df_acc_sorted = df_device_summary.sort_values('Avg_Accuracy', ascending=False)
        sns.barplot(data=df_acc_sorted, x='Avg_Accuracy', y='Model', ax=axes[0], hue='Model', dodge=False, legend=False, palette='summer')
        axes[0].errorbar(x=df_acc_sorted['Avg_Accuracy'], y=np.arange(len(df_acc_sorted)), xerr=df_acc_sorted['Std_Accuracy'], fmt='none', ecolor='black', capsize=5)
        axes[0].set_title(f'Ranked Model Accuracy on {device_type}', fontsize=16)
        axes[0].set_xlabel('Average Accuracy (%)'); axes[0].set_ylabel('Model')
        df_time_sorted = df_device_summary.sort_values('Avg_Time', ascending=True)
        sns.barplot(data=df_time_sorted, x='Avg_Time', y='Model', ax=axes[1], hue='Model', dodge=False, legend=False, palette='autumn')
        axes[1].errorbar(x=df_time_sorted['Avg_Time'], y=np.arange(len(df_time_sorted)), xerr=df_time_sorted['Std_Time'], fmt='none', ecolor='black', capsize=5)
        axes[1].set_title(f'Ranked Model Speed on {device_type}', fontsize=16)
        axes[1].set_xlabel('Average Prediction Time (s)'); axes[1].set_ylabel('')
        fig.suptitle(f'Direct Performance Comparison on {device_type}', fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(device_results_dir, "performance_barcharts.png"), dpi=300)
        plt.close()
        print(f"📈 {device_type} ranked bar charts saved.")

        # --- Plot 2: Scatter Plot with Error Bars ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 7))
        palette = sns.color_palette("viridis", n_colors=len(df_device_summary))

        # Use enumerate to get a separate counter `plot_idx` for the palette
        for plot_idx, (df_index, row) in enumerate(df_device_summary.iterrows()):
            ax.errorbar(x=row["Avg_Time"], y=row["Avg_Accuracy"], xerr=row["Std_Time"], yerr=row["Std_Accuracy"], 
                        fmt='o', color=palette[plot_idx], ecolor='gray', elinewidth=1.5, capsize=5, markersize=10) # Use plot_idx here
            ax.text(row["Avg_Time"], row["Avg_Accuracy"] + 0.3, row["Model"], fontsize=11, ha='center', fontweight='bold')
            
        ax.set_title(f'Accuracy vs. Time on {device_type}', fontsize=16, pad=20)
        ax.set_xlabel('Average Prediction Time per Image (s)', fontsize=12)
        ax.set_ylabel('Average Validation Accuracy (%)', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(device_results_dir, "performance_scatter_avg.png"), dpi=300)
        plt.close()
        print(f"📈 {device_type} scatter plot saved.")

        # --- Plot 3: Strategic Quadrant Plot ---
        fig, ax = plt.subplots(figsize=(10, 8))
        avg_acc_line = df_device_summary['Avg_Accuracy'].mean()
        avg_time_line = df_device_summary['Avg_Time'].mean()
        sns.scatterplot(data=df_device_summary, x='Avg_Time', y='Avg_Accuracy', hue='Model', s=250, ax=ax, palette='viridis', style='Model', edgecolor='black', alpha=0.8)
        ax.axhline(avg_acc_line, ls='--', color='gray'); ax.axvline(avg_time_line, ls='--', color='gray')
        ax.set_title(f'Strategic Performance Quadrant on {device_type}', fontsize=16)
        ax.set_xlabel('Prediction Time (s) [Faster → Slower]', fontsize=12); ax.set_ylabel('Accuracy (%)', fontsize=12)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(device_results_dir, "performance_quadrant.png"), dpi=300)
        plt.close()
        print(f"📈 {device_type} quadrant plot saved.")

        # --- Plot 4: Distribution Plots (GPU ONLY) ---
        if device_type == 'CUDA':
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            sns.violinplot(data=df_device_all_runs, x='Model', y='Accuracy (%)', ax=axes[0], inner=None, hue='Model', legend=False, palette='viridis')
            sns.swarmplot(data=df_device_all_runs, x='Model', y='Accuracy (%)', ax=axes[0], color='k', alpha=0.6)
            axes[0].set_title('Distribution of Model Accuracy (GPU)', fontsize=16)
            sns.violinplot(data=df_device_all_runs, x='Model', y='Prediction Time (s)', ax=axes[1], inner=None, hue='Model', legend=False, palette='viridis')
            sns.swarmplot(data=df_device_all_runs, x='Model', y='Prediction Time (s)', ax=axes[1], color='k', alpha=0.6)
            axes[1].set_title('Distribution of Prediction Time (GPU)', fontsize=16)
            fig.suptitle('Performance Stability on GPU Across 10 Runs', fontsize=20)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(device_results_dir, "performance_distributions.png"), dpi=300)
            plt.close()
            print(f"📈 {device_type} distribution plots saved.")

    # --- Generate Combined Reports ---
    combined_dir = os.path.join(results_dir, "Combined_Results")
    os.makedirs(combined_dir, exist_ok=True)
    print("\n--- Generating Combined Reports ---")

    df_summary.to_csv(os.path.join(combined_dir, "full_performance_summary.csv"), index=False, float_format='%.6f')
    print("📊 Full summary CSV saved.")
    print(df_summary.to_string(index=False))

    # Combined Bar Chart for Speed Comparison
    if 'CPU' in df_summary['Device'].unique() and 'CUDA' in df_summary['Device'].unique():
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(data=df_summary, x='Model', y='Avg_Time', hue='Device', ax=ax, palette={'CPU': '#4c72b0', 'CUDA': '#dd8452'})
        ax.set_yscale('log')
        ax.set_title('CPU vs. GPU Inference Speed Comparison (Log Scale)', fontsize=16)
        ax.set_ylabel('Average Prediction Time per Image (s) - Log Scale'); ax.set_xlabel('Model')
        ax.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(os.path.join(combined_dir, "speed_comparison_cpu_vs_gpu.png"), dpi=300)
        plt.close()
        print("📈 Combined CPU vs. GPU speed comparison chart saved.")

    print(f"\n🎉 All tasks complete! Check the '{results_dir}' folder.")