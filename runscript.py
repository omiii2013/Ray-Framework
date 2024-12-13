import os

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the dataset
dataset_path = os.path.join(current_dir, 'Datasets', 'breast_cancer.csv')

# # Print the dataset path (optional, for debugging)
# print(f"Dataset Path: {dataset_path}")

# Define a custom log directory
custom_log_dir = current_dir +"\\ray_logs"

print(custom_log_dir)