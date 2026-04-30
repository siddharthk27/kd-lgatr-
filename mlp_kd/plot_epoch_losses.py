import re
import matplotlib.pyplot as plt
from collections import defaultdict

log_file_path = "/home/jay_agarwal_2022/kd-lgatr/mlp_kd/training.log"
output_plot_path = "/home/jay_agarwal_2022/kd-lgatr/mlp_kd/epoch_loss_curve.png"

def main():
    print(f"Parsing log file {log_file_path}...")
    
    # regex for: Epoch 1 | Batch 100/9461 | Loss: 4.8520
    pattern = re.compile(r"Epoch\s+(\d+)\s+\|\s+Batch\s+(\d+)/\d+\s+\|\s+Loss:\s+([\d.]+)")
    
    epoch_losses = defaultdict(list)
    
    with open(log_file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(3))
                epoch_losses[epoch].append(loss)
                
    if not epoch_losses:
        print("No training data found in the log file.")
        return
        
    epochs = sorted(epoch_losses.keys())
    # Estimate average epoch loss by averaging the sampled batch losses
    estimated_epoch_avg_losses = [sum(epoch_losses[e]) / len(epoch_losses[e]) for e in epochs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, estimated_epoch_avg_losses, marker='o', label='Estimated Average Loss', color='blue')
    plt.title('Estimated Average Epoch Loss\n(Averaged over sampled batches)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_plot_path)
    print(f"Plot saved to {output_plot_path}")

if __name__ == "__main__":
    main()
