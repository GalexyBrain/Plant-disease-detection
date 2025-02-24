import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
csv_file = 'training_log.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file)

# Check the columns to ensure we are using the correct ones (optional)
print(df.columns)

# Plot the graphs
plt.figure(figsize=(18, 6))

# Plot Accuracy (Training and Validation)
plt.subplot(1, 3, 1)
plt.plot(df['epoch'], df['accuracy'], label='Training Accuracy', color='blue')
plt.plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy', color='green', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='best')

# Plot Loss (Training and Validation)
plt.subplot(1, 3, 2)
plt.plot(df['epoch'], df['loss'], label='Training Loss', color='red')
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='orange', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='best')

# Plot Learning Rate
plt.subplot(1, 3, 3)
plt.plot(df['epoch'], df['learning_rate'], label='Learning Rate', color='purple')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Over Epochs')
plt.legend(loc='best')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot as an image
plt.savefig('accuracy_loss_lr_plot.png')

# Show the plot
plt.show()
