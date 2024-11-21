# utils/engine.py
import torch
from tqdm import tqdm

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, config):
    """
    Trains the model for one epoch and returns the average loss, aggregated debug metrics, and the number of failed batches.

    Args:
        model (torch.nn.Module): The model to train.
        criterion (callable): The loss function.
        data_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        config (dict): Configuration dictionary containing training parameters.

    Returns:
        tuple: (average_loss, aggregated_debug_metrics, batch_failed)
    """
    batch_failed = 0
    model.train()
    # If the criterion has a train mode, enable it. Otherwise, this line can be removed.
    if hasattr(criterion, 'train'):
        criterion.train()
    optimizer.zero_grad()

    accum_steps = config.get('gradient_accumulation_steps', 1)
    scaler = torch.cuda.amp.GradScaler()
    loop = tqdm(data_loader, total=len(data_loader))

    total_loss = 0.0
    debug_metrics_sum = {}
    debug_metrics_count = {}
    valid_steps = 0

    for i, data_dict in enumerate(loop):
        samples, targets = data_dict

        samples = samples.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        shape = samples.shape  # For error logging

        try:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss_dict, debuginfo = criterion(outputs, targets)
                loss = sum(loss_dict.values()) / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(data_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            # Accumulate loss
            total_loss += loss.item() * accum_steps  # Multiply back to get actual loss

            # Accumulate debug metrics
            for key, value in debuginfo.items():
                if key not in debug_metrics_sum:
                    debug_metrics_sum[key] = 0.0
                    debug_metrics_count[key] = 0
                debug_metrics_sum[key] += value.item()
                debug_metrics_count[key] += 1

            valid_steps += 1

            # Compute average metrics for display
            avg_loss = total_loss / valid_steps
            avg_metrics = {key: debug_metrics_sum[key] / debug_metrics_count[key] 
                           for key in debug_metrics_sum}

            loop.set_postfix(loss=avg_loss,pit_acc=avg_metrics['pit_acc'],InstAcc=avg_metrics['InstAcc'],
                             R_M_ST=avg_metrics['R_M_ST'],R_M_dur=avg_metrics['R_M_dur'],R_M_v=avg_metrics['R_M_v'])

        except Exception as e:
            print(f"Error in training: {e}")
            print(f"SAMPLES shape: {shape}")
            batch_failed += 1

    # Compute final averages
    avg_loss = total_loss / valid_steps if valid_steps > 0 else 0.0
    aggregated_debug_metrics = {key: debug_metrics_sum[key] / debug_metrics_count[key] 
                                 for key in debug_metrics_sum}

    return avg_loss, aggregated_debug_metrics, batch_failed

def evaluate(model, criterion, data_loader, device):
    """
    Evaluates the model on the validation/test set and returns the average loss, aggregated debug metrics, and the number of failed batches.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion (callable): The loss function.
        data_loader (DataLoader): DataLoader for validation/test data.
        device (torch.device): Device to run the evaluation on.

    Returns:
        tuple: (average_loss, aggregated_debug_metrics, batch_failed)
    """
    batch_failed = 0
    model.eval()
    # If the criterion has an eval mode, enable it. Otherwise, this line can be removed.
    if hasattr(criterion, 'eval'):
        criterion.eval()

    total_loss = 0.0
    debug_metrics_sum = {}
    debug_metrics_count = {}
    valid_steps = 0

    with torch.no_grad():
        loop = tqdm(data_loader, total=len(data_loader), desc='Eval')
        for data_dict in loop:
            samples, targets = data_dict
            samples = samples.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            shape = samples.shape  # For error logging

            try:
                with torch.cuda.amp.autocast():
                    outputs = model(samples)
                    loss_dict, debuginfo = criterion(outputs, targets)
                    loss = sum(loss_dict.values())

                # Accumulate loss
                total_loss += loss.item()

                # Accumulate debug metrics
                for key, value in debuginfo.items():
                    if key not in debug_metrics_sum:
                        debug_metrics_sum[key] = 0.0
                        debug_metrics_count[key] = 0
                    debug_metrics_sum[key] += value.item()
                    debug_metrics_count[key] += 1

                valid_steps += 1

                # Compute average metrics for display
                avg_loss = total_loss / valid_steps
                avg_metrics = {key: debug_metrics_sum[key] / debug_metrics_count[key] 
                               for key in debug_metrics_sum}

                loop.set_postfix(loss=avg_loss,pit_acc=avg_metrics['pit_acc'],InstAcc=avg_metrics['InstAcc'],
                             R_M_ST=avg_metrics['R_M_ST'],R_M_dur=avg_metrics['R_M_dur'])

            except Exception as e:
                print(f"Error in evaluation: {e}")
                print(f"SAMPLES shape: {shape}")
                batch_failed += 1

    # Compute final averages
    avg_loss = total_loss / valid_steps if valid_steps > 0 else 0.0
    aggregated_debug_metrics = {key: debug_metrics_sum[key] / debug_metrics_count[key] 
                                 for key in debug_metrics_sum}
    for key, value in aggregated_debug_metrics.items():
        aggregated_debug_metrics[key] = value

    return avg_loss, aggregated_debug_metrics, batch_failed
