# train.py
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from utils.dataset import AudioDataset
from models.detr_audio import DETRAudio
from utils.engine import train_one_epoch, evaluate
from utils.criterion import CustomCriterion
from torchaudio.transforms import MelSpectrogram, Spectrogram
import os
import csv
from utils.utils import save_tensor_as_png, get_latest_checkpoint
import json5 as json

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys  # Added for interactivity checks
import argparse  # Added for command-line arguments

def main(CONFIG):
    device = CONFIG['device']
    version = CONFIG.get('version', 'v0')
    save_by_spoch = CONFIG.get('save_by_spoch', 1)

    # Set seed for reproducibility if defined in config debug
    if CONFIG.get('debug', True):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import random
        random.seed(42)

    # Create logs directory if it doesn't exist
    logs_dir = CONFIG.get('logs_dir', 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Define paths for training and evaluation metrics
    train_metrics_path = os.path.join(logs_dir, f'{version}_train_metrics.csv')
    eval_metrics_path = os.path.join(logs_dir, f'{version}_eval_metrics.csv')
    config_save_path = os.path.join(logs_dir, f'{version}_config.json')

    # Initialize CSV files with headers if they don't exist
    if not os.path.exists(train_metrics_path):
        with open(train_metrics_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
	    # TODO: Updated header to include all metric keys
            writer.writerow(header)
        print(f"Created training metrics file at {train_metrics_path}")

    if not os.path.exists(eval_metrics_path):
        with open(eval_metrics_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # TODO: Updated header to include all metric keys
            writer.writerow(header)
        print(f"Created evaluation metrics file at {eval_metrics_path}")

    # Save initial CONFIG if not already saved
    if not os.path.exists(config_save_path):
        with open(config_save_path, 'w') as f:
            json.dump(CONFIG, f, indent=4)
        print(f"Created config file at {config_save_path}")

    # Datasets and DataLoaders
    print("Loading data from directory:", CONFIG['data_dir'])

    all_audio_files = [f for f in os.listdir(CONFIG['data_dir']) if f.endswith('.wav')]

    # Initialize train and validation lists
    train_audio_files = []
    val_audio_files = []
    start_epoch = 0

    model_save_dir = CONFIG['model_save_path']
    latest_checkpoint, start_epoch = get_latest_checkpoint(model_save_dir, version)
    model = DETRAudio(CONFIG).to(device)
    model_config_path = os.path.join(logs_dir, f'{version}_model_config.txt')  # Changed extension for clarity
    if not os.path.exists(model_config_path):
        with open(model_config_path, 'w') as f:
            f.write(str(model))
    criterion = CustomCriterion(CONFIG).to(device)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG['lr'], 
        weight_decay=CONFIG['weight_decay']
    )

    # Create scheduler based on CONFIG
    scheduler_type = CONFIG.get('scheduler_type', None)
    if scheduler_type is not None:
        scheduler_params = CONFIG.get('scheduler_params', {})
        if scheduler_type == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        elif scheduler_type == 'ExponentialLR':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
        elif scheduler_type == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
        # Add more schedulers if needed
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}")
    else:
        scheduler = None

    # Initialize freezing configuration
    freeze_config = {
        "freeze_backbone": False,
        "freeze_transformer": False,
        "freeze_heads": False
    }

    # Optionally resume from a checkpoint
    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

        # Prompt for new learning rate
        if 'y' in input("Enter new learning rate? (y/n): ").lower():
            try:
                new_lr = float(input("Enter new learning rate: "))
                optimizer.param_groups[0]['lr'] = new_lr
                print(f"Learning rate updated to {new_lr}")
            except ValueError:
                print("Invalid input. Learning rate not changed.")
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint: {latest_checkpoint}, Epoch {checkpoint['epoch']}")

        # Load the split from the checkpoint
        train_audio_files = checkpoint.get('train_audio_files', [])
        val_audio_files = checkpoint.get('val_audio_files', [])
        if not train_audio_files or not val_audio_files:
            raise ValueError("Checkpoint does not contain train and validation split information.")

        # Load scheduler state dict
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Handle freezing layers based on command-line flags and interactivity
        if any([CONFIG.get('freeze_backbone', False),
                CONFIG.get('freeze_transformer', False),
                CONFIG.get('freeze_heads', False),
                CONFIG.get('freeze_all', False)]):
            if CONFIG.get('freeze_all', False):
                freeze_config = {
                    "freeze_backbone": True,
                    "freeze_transformer": True,
                    "freeze_heads": True
                }
                print("Freezing all layers as per --freeze_all flag.")
            else:
                freeze_config["freeze_backbone"] = CONFIG.get('freeze_backbone', False)
                freeze_config["freeze_transformer"] = CONFIG.get('freeze_transformer', False)
                freeze_config["freeze_heads"] = CONFIG.get('freeze_heads', False)
                print(f"Freezing layers as per flags: {freeze_config}")
            model.freeze_layers(freeze_config)

            # **Add "log" key to CONFIG with freeze information**
            CONFIG['log'] = freeze_config
            # **Save the updated CONFIG with "log"**
            try:
                with open(config_save_path, 'w') as f:
                    json.dump(CONFIG, f, indent=4)
                print(f"Updated config file with freeze information at {config_save_path}")
            except Exception as e:
                print(f"Error updating config file with log: {e}")

        elif not CONFIG.get('no_prompt', False) and sys.stdin.isatty():
            # Interactive prompting
            print("Do you want to freeze specific layers of the model?")
            freeze_backbone = input("Freeze backbone? (y/n): ").strip().lower() == 'y'
            freeze_transformer = input("Freeze transformer? (y/n): ").strip().lower() == 'y'
            freeze_heads = input("Freeze heads? (y/n): ").strip().lower() == 'y'
            freeze_config["freeze_backbone"] = freeze_backbone
            freeze_config["freeze_transformer"] = freeze_transformer
            freeze_config["freeze_heads"] = freeze_heads
            if any(freeze_config.values()):
                model.freeze_layers(freeze_config)
                print(f"Applied freezing configuration: {freeze_config}")

                # **Add "log" key to CONFIG with freeze information**
                CONFIG['log'] = freeze_config
                # **Save the updated CONFIG with "log"**
                try:
                    with open(config_save_path, 'w') as f:
                        json.dump(CONFIG, f, indent=4)
                    print(f"Updated config file with freeze information at {config_save_path}")
                except Exception as e:
                    print(f"Error updating config file with log: {e}")
            else:
                print("No layers were frozen.")
        else:
            # Non-interactive and no freezing flags set
            print("Skipping freezing layers (non-interactive and no freezing flags provided).")
    else:
        start_epoch = 1  # Start from epoch 1 if no checkpoint is found
        print("No checkpoint found. Starting training from scratch.")
        import random
        random.shuffle(all_audio_files)
        split_ratio = CONFIG.get('split_ratio', 0.8)
        split_index = int(len(all_audio_files) * split_ratio)
        train_audio_files = all_audio_files[:split_index]
        val_audio_files = all_audio_files[split_index:]

    train_dataset = AudioDataset(
        train_audio_files, 
        CONFIG['cache_dir'], 
        split='train', 

        config=CONFIG,

    )
    val_dataset = AudioDataset(
        val_audio_files, 
        CONFIG['cache_dir'], 
        split='val', 

        config=CONFIG,

    )
    if CONFIG.get('debug', False):
        train_loader = DataLoader(
            train_dataset, 
            batch_size=CONFIG['batch_size'], 
            shuffle=True, 
            num_workers=CONFIG['num_workers'],
            collate_fn=train_dataset.collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=CONFIG['batch_size'], 
            shuffle=True, 
            num_workers=CONFIG['num_workers'],
            collate_fn=val_dataset.collate_fn
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=CONFIG['batch_size'], 
            shuffle=True, 
            num_workers=CONFIG['num_workers'],
            collate_fn=train_dataset.collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=CONFIG['batch_size'], 
            shuffle=True, 
            num_workers=CONFIG['num_workers'],
            collate_fn=val_dataset.collate_fn
        )
        

    def save_checkpoint(model_save_dir, version, epoch, model, optimizer, scheduler, CONFIG, train_audio_files, val_audio_files):
        """
        Saves the model checkpoint with the specified version and epoch, including train and validation splits.

        Args:
            model_save_dir (str): Directory where model checkpoints are saved.
            version (str): Current version identifier.
            epoch (int): Current epoch number.
            model (torch.nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer to save.
            scheduler (torch.optim.lr_scheduler): The scheduler to save.
            CONFIG (dict): Configuration dictionary.
            train_audio_files (list): List of training audio filenames.
            val_audio_files (list): List of validation audio filenames.
        """
        checkpoint_path = os.path.join(model_save_dir, f"{version}_model_e{epoch}.pth")
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'config': CONFIG,
                'train_audio_files': train_audio_files,
                'val_audio_files': val_audio_files
            }, checkpoint_path)
            print(f"Saved checkpoint at {checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    # Training Loop
    for epoch in range(start_epoch, CONFIG['epochs'] + 1):
        print(f"\n--- Epoch {epoch}/{CONFIG['epochs']} ---")

        # Train for one epoch
        avg_train_loss, aggregated_debug_metrics, batch_failed = train_one_epoch(
            model, 
            criterion, 
            train_loader, 
            optimizer, 
            device, 
            epoch, 
            CONFIG
        )
        print(f"Training   - Epoch: {epoch}, Loss: {avg_train_loss:.4f}")
        for key, value in aggregated_debug_metrics.items():
            print(f"            - {key}: {value}")

        # Extract metrics safely with default values
        pit_acc = aggregated_debug_metrics.get('pit_acc', 0.0)
        InstAcc = aggregated_debug_metrics.get('InstAcc', 0.0)
        R_M_ST = aggregated_debug_metrics.get('R_M_ST', 0.0)
        R_M_dur = aggregated_debug_metrics.get('R_M_dur', 0.0)
        R_M_v = aggregated_debug_metrics.get('R_M_v', 0.0)

        # Write training metrics to CSV
        try:
            with open(train_metrics_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    epoch, 
                    f"{avg_train_loss:.4f}", 
                    f"{pit_acc:.4f}", 
                    f"{InstAcc:.4f}", 
                    f"{R_M_ST:.4f}", 
                    f"{R_M_dur:.4f}", 
                    f"{R_M_v:.4f}",
                    batch_failed,
                    optimizer.param_groups[0]['lr']
                ])
            print(f"Appended training metrics to {train_metrics_path}")
        except Exception as e:
            print(f"Error writing to training metrics file: {e}")

        # Evaluate on validation every 5 epochs
        def eval():
            avg_val_loss, aggregated_val_metrics, val_batch_failed = evaluate(
                model, 
                criterion, 
                val_loader, 
                device
            )
            print(f"Validation - Epoch: {epoch}, Loss: {avg_val_loss:.4f}")
            for key, value in aggregated_val_metrics.items():
                print(f"            - {key}: {value}")

            # Extract validation metrics safely with default values
            pit_acc_val = aggregated_val_metrics.get('pit_acc', 0.0)
            InstAcc_val = aggregated_val_metrics.get('InstAcc', 0.0)
            R_M_ST_val = aggregated_val_metrics.get('R_M_ST', 0.0)
            R_M_dur_val = aggregated_val_metrics.get('R_M_dur', 0.0)
            R_M_v_val = aggregated_val_metrics.get('R_M_v', 0.0)

            # Write validation metrics to CSV
            try:
                with open(eval_metrics_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        epoch, 
                        f"{avg_val_loss:.4f}", 
                        f"{pit_acc_val:.4f}", 
                        f"{InstAcc_val:.4f}", 
                        f"{R_M_ST_val:.4f}", 
                        f"{R_M_dur_val:.4f}",
                        f"{R_M_v_val:.4f}",
                        val_batch_failed,
                        optimizer.param_groups[0]['lr']
                    ])
                print(f"Appended evaluation metrics to {eval_metrics_path}")
            except Exception as e:
                print(f"Error writing to evaluation metrics file: {e}")
            return avg_val_loss, aggregated_val_metrics, val_batch_failed

        # Update the scheduler
        if scheduler is not None:
            if scheduler_type == 'ReduceLROnPlateau':
                avg_val_loss, _, _ = eval()

                # Use validation loss as metric
                scheduler.step(avg_val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr}")
            else:
                if epoch % save_by_spoch == 0:
                    avg_val_loss, _, _ = eval()
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr}")
        else:
            if epoch % save_by_spoch  == 0:
                avg_val_loss, _, _ = eval()

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr}")

        # Save model checkpoint every 5 epochs
        if epoch % save_by_spoch  == 0:
            save_checkpoint(
                model_save_dir=model_save_dir,
                version=version,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                CONFIG=CONFIG,
                train_audio_files=train_audio_files,
                val_audio_files=val_audio_files
            )

    print("\nTraining complete.")

if __name__ == '__main__':

    # Get argument containing config file
    parser = argparse.ArgumentParser(description="Train DETRAudio Model with Optional Freezing of Layers")
    parser.add_argument('--config', type=str, help='Config file path', default='config.json')

    # Added command-line flags for freezing layers
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze the backbone layers.')
    parser.add_argument('--freeze_transformer', action='store_true', help='Freeze the transformer layers.')
    parser.add_argument('--freeze_heads', action='store_true', help='Freeze the classification and regression heads.')
    parser.add_argument('--freeze_all', action='store_true', help='Freeze all layers (backbone, transformer, heads).')
    parser.add_argument('--no_prompt', action='store_true', help='Do not prompt for freezing layers when resuming training.')

    args = parser.parse_args()
    with open(args.config) as f:
        CONFIG = dict(json.load(f))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    CONFIG['device'] = device

    # Add freeze flags to CONFIG
    CONFIG['freeze_backbone'] = args.freeze_backbone
    CONFIG['freeze_transformer'] = args.freeze_transformer
    CONFIG['freeze_heads'] = args.freeze_heads
    CONFIG['freeze_all'] = args.freeze_all
    CONFIG['no_prompt'] = args.no_prompt

    main(CONFIG)
