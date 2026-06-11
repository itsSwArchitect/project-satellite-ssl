
import torch
import csv
from pathlib import Path
import numpy as np

class AverageMeter:
    """Track running average"""
    
    def __init__(self, name='Loss'):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"

class ExperimentLogger:
    """Log metrics to CSV"""
    
    def __init__(self, log_dir="./logs", experiment_name="experiment"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{experiment_name}_metrics.csv"
        self.metrics_history = []
    
    def log_epoch(self, epoch, metrics):
        metrics["epoch"] = epoch
        self.metrics_history.append(metrics)
        
        # Write to CSV
        mode = "w" if epoch == 0 else "a"
        with open(self.log_file, mode, newline="") as f:
            if epoch == 0:
                writer = csv.DictWriter(f, fieldnames=metrics.keys())
                writer.writeheader()
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writerow(metrics)
        
        # Print
        log_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k != "epoch"])
        print(f"Epoch {epoch+1}: {log_str}")
    
    def plot_metrics(self):
        return self.metrics_history

def save_checkpoint(model, optimizer, epoch, checkpoint_dir="./checkpoints", name="checkpoint"):
    """Save model checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"{name}_epoch_{epoch}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)
    
    return checkpoint_path

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
