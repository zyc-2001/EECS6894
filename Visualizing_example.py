import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('training_results.csv')

# Get all the different model names
models = data['Model'].unique()

# Plot the figures
plt.figure(figsize=(12, 5))

# Plot the accuracy curves
plt.subplot(1, 2, 1)
for model in models:
    model_data = data[data['Model'] == model]
    plt.plot(model_data['Epoch'], model_data['Train Accuracy'], label=f'{model} - Train', marker='o')
    plt.plot(model_data['Epoch'], model_data['Validation Accuracy'], label=f'{model} - Validation', marker='o')

plt.title('Training and Validation Accuracy for Different Models')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot the loss curves
plt.subplot(1, 2, 2)
for model in models:
    model_data = data[data['Model'] == model]
    plt.plot(model_data['Epoch'], model_data['Train Loss'], label=f'{model} - Train', marker='o')
    plt.plot(model_data['Epoch'], model_data['Validation Loss'], label=f'{model} - Validation', marker='o')

plt.title('Training and Validation Loss for Different Models')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Adjust the layout and show the plot
plt.tight_layout()
plt.show()