import re
import matplotlib.pyplot as plt

log_file_path = "/home/jay_agarwal_2022/kd-lgatr/mlp_kd_engd/ml_kd_engd_depth6.log"
output_plot_path = "/home/jay_agarwal_2022/kd-lgatr/mlp_kd_engd/training_loss_curve_depth6.png"

def parse_log_file(file_path):
    batch_soft_kd_losses = []
    batch_hard_losses = []
    batch_total_losses = []
    epoch_avg_kd_losses = []
    
    batch_pattern = re.compile(r"Epoch\s+(\d+)/\d+\s+\|\s+Batch\s+(\d+)/\d+\s+\|\s+Soft KD Loss:\s+([\d.]+)\s+\|\s+Hard Loss:\s+([\d.]+)\s+\|\s+Total Loss:\s+([\d.]+)")
    avg_pattern = re.compile(r"====>\s+Epoch\s+(\d+)\s+Average KD Loss:\s+([\d.]+)")
    
    with open(file_path, 'r') as f:
        for line in f:
            batch_match = batch_pattern.search(line)
            if batch_match:
                soft_kd_loss = float(batch_match.group(3))
                hard_loss = float(batch_match.group(4))
                total_loss = float(batch_match.group(5))
                batch_soft_kd_losses.append(soft_kd_loss)
                batch_hard_losses.append(hard_loss)
                batch_total_losses.append(total_loss)
                continue
                
            avg_match = avg_pattern.search(line)
            if avg_match:
                epoch = int(avg_match.group(1))
                avg_kd_loss = float(avg_match.group(2))
                epoch_avg_kd_losses.append(avg_kd_loss)
                
    return batch_soft_kd_losses, batch_hard_losses, batch_total_losses, epoch_avg_kd_losses

def main():
    print(f"Parsing log file {log_file_path}...")
    batch_soft_kd_losses, batch_hard_losses, batch_total_losses, epoch_avg_kd_losses = parse_log_file(log_file_path)
    
    if not batch_soft_kd_losses:
        print("No training data found in the log file.")
        return
        
    print(f"Parsed {len(batch_soft_kd_losses)} batch records and {len(epoch_avg_kd_losses)} epoch average records.")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Batch-level losses
    window_size = max(1, len(batch_soft_kd_losses) // 200)
    
    def moving_average(data, w):
        if w <= 1:
            return data
        import numpy as np
        return np.convolve(data, np.ones(w), 'valid') / w
        
    smooth_soft_kd = moving_average(batch_soft_kd_losses, window_size)
    smooth_hard = moving_average(batch_hard_losses, window_size)
    smooth_total = moving_average(batch_total_losses, window_size)
    
    x_smooth = range(window_size - 1, len(batch_soft_kd_losses))
    
    ax1.plot(x_smooth, smooth_soft_kd, label='Sequential Batch Soft KD Loss (Smoothed)', alpha=0.8)
    ax1.plot(x_smooth, smooth_hard, label='Sequential Batch Hard Loss (Smoothed)', alpha=0.8)
    ax1.plot(x_smooth, smooth_total, label='Sequential Batch Total Loss (Smoothed)', alpha=0.8, color='purple')
    ax1.set_title('Training Losses over Batches (Depth 6)')
    ax1.set_xlabel('Batch Step')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Epoch-level average total loss
    epochs = range(1, len(epoch_avg_kd_losses) + 1)
    ax2.plot(epochs, epoch_avg_kd_losses, marker='o', label='Average KD Loss', color='green')
    ax2.set_title('Average Epoch KD Loss (Depth 6)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_plot_path)
    print(f"Plot saved to {output_plot_path}")

if __name__ == "__main__":
    main()
