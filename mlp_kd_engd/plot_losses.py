import re
import matplotlib.pyplot as plt

log_file_path = "/home/jay_agarwal_2022/kd-lgatr/mlp_kd_engd/ml_kd_engd.log"
output_plot_path = "/home/jay_agarwal_2022/kd-lgatr/mlp_kd_engd/training_loss_curve.png"

def parse_log_file(file_path):
    batch_kd_losses = []
    batch_base_losses = []
    epoch_avg_kd_losses = []
    
    batch_pattern = re.compile(r"Epoch\s+(\d+)/\d+\s+\|\s+Batch\s+(\d+)/\d+\s+\|\s+KD Loss:\s+([\d.]+)\s+\|\s+Base Loss:\s+([\d.]+)")
    avg_pattern = re.compile(r"====>\s+Epoch\s+(\d+)\s+Average KD Loss:\s+([\d.]+)")
    
    with open(file_path, 'r') as f:
        for line in f:
            batch_match = batch_pattern.search(line)
            if batch_match:
                # We can just store sequential batches
                kd_loss = float(batch_match.group(3))
                base_loss = float(batch_match.group(4))
                batch_kd_losses.append(kd_loss)
                batch_base_losses.append(base_loss)
                continue
                
            avg_match = avg_pattern.search(line)
            if avg_match:
                epoch = int(avg_match.group(1))
                avg_kd_loss = float(avg_match.group(2))
                epoch_avg_kd_losses.append(avg_kd_loss)
                
    return batch_kd_losses, batch_base_losses, epoch_avg_kd_losses

def main():
    print(f"Parsing log file {log_file_path}...")
    batch_kd_losses, batch_base_losses, epoch_avg_kd_losses = parse_log_file(log_file_path)
    
    if not batch_kd_losses:
        print("No training data found in the log file.")
        return
        
    print(f"Parsed {len(batch_kd_losses)} batch records and {len(epoch_avg_kd_losses)} epoch average records.")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Batch-level losses
    # Using a moving average for readability if there are many batches
    window_size = max(1, len(batch_kd_losses) // 200)
    
    def moving_average(data, w):
        if w <= 1:
            return data
        import numpy as np
        return np.convolve(data, np.ones(w), 'valid') / w
        
    smooth_kd = moving_average(batch_kd_losses, window_size)
    smooth_base = moving_average(batch_base_losses, window_size)
    
    x_smooth = range(window_size - 1, len(batch_kd_losses))
    
    ax1.plot(x_smooth, smooth_kd, label='Sequential Batch KD Loss (Smoothed)', alpha=0.8)
    ax1.plot(x_smooth, smooth_base, label='Sequential Batch Base Loss (Smoothed)', alpha=0.8)
    ax1.set_title('Training Losses over Batches')
    ax1.set_xlabel('Batch Step')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Epoch-level average KD loss
    epochs = range(1, len(epoch_avg_kd_losses) + 1)
    ax2.plot(epochs, epoch_avg_kd_losses, marker='o', label='Average KD Loss', color='green')
    ax2.set_title('Average Epoch KD Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_plot_path)
    print(f"Plot saved to {output_plot_path}")

if __name__ == "__main__":
    main()
