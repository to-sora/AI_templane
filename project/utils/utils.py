import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import os
import torch
import glob
import re

# Example usage:
# save_tensor_as_png("path_to_your_tensor.pt")
def get_latest_checkpoint(model_save_dir, version):
    """
    Finds the latest checkpoint file for the given version.

    Args:
        model_save_dir (str): Directory where model checkpoints are saved.
        version (str): Current version identifier.

    Returns:
        tuple: (latest_checkpoint_path (str or None), latest_epoch (int))
    """
    pattern = os.path.join(model_save_dir, f"{version}_model_e*.pth")
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        return None, 0  # No checkpoint found
    
    # Extract epoch numbers using regex
    epoch_pattern = re.compile(rf"{re.escape(version)}_model_e(\d+)\.pth$")
    epochs = []
    for file in checkpoint_files:
        basename = os.path.basename(file)
        match = epoch_pattern.match(basename)
        if match:
            epochs.append(int(match.group(1)))
    
    if not epochs:
        return None, 0
    
    latest_epoch = max(epochs)
    latest_checkpoint = os.path.join(model_save_dir, f"{version}_model_e{latest_epoch}.pth")
    return latest_checkpoint, latest_epoch
