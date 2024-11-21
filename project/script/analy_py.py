import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Specify the directory containing .pt files
pt_directory  = "data/cache/"   # Replace with your directory path

# Initialize lists to store values
last_dim_lengths = []
average_values = []
max_values = []

# Iterate through all files in the directory
for filename in tqdm(os.listdir(pt_directory)):
    if filename.endswith(".pt"):
        filepath = os.path.join(pt_directory, filename)
        
        # Load the tensor
        tensor = torch.load(filepath)
        
        # Ensure the tensor has at least one dimension
        if tensor.ndim > 0:
            # Extract the length of the last dimension
            last_dim_lengths.append(tensor.shape[-1])
            # Compute average value
            average_values.append(tensor.mean().item())
            # Compute max value
            max_values.append(tensor.max().item())

# Plotting Last Dimension Length Distribution
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.hist(last_dim_lengths, bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Length of Last Dimension')
plt.ylabel('Count')
plt.title('Last Dimension Length Distribution')

# Plotting Average Value Distribution
plt.subplot(1, 3, 2)
plt.hist(average_values, bins=10, color='lightgreen', edgecolor='black')
plt.xlabel('Average Value')
plt.ylabel('Count')
plt.title('Average Value Distribution')

# Plotting Max Value Distribution
plt.subplot(1, 3, 3)
plt.hist(max_values, bins=10, color='salmon', edgecolor='black')
plt.xlabel('Max Value')
plt.ylabel('Count')
plt.title('Max Value Distribution')

plt.tight_layout()
plt.show()
plt.savefig("tensor_stats.png")
