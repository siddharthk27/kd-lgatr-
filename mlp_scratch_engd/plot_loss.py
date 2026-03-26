import re
import matplotlib.pyplot as plt

def plot_loss_curve(log_file_path):
    epochs = []
    losses = []

    # Regex pattern to match the epoch average loss summary lines
    # Example target: "====> Epoch 1 Average Loss: 0.2929 | LR: 0.000800"
    pattern = re.compile(r'====>\s*Epoch\s*(\d+)\s*Average Loss:\s*([\d.]+)')

    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    # Extract the epoch number and the loss value
                    epoch = int(match.group(1))
                    loss = float(match.group(2))
                    
                    epochs.append(epoch)
                    losses.append(loss)
                    
        if not epochs:
            print("No training data found. Please check the log file format.")
            return

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, marker='.', linestyle='-', color='b', linewidth=1.5, label='Epoch Average Loss')
        
        # Formatting the plot
        plt.title('Training Loss Curve', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Average Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig('loss_curve.png', dpi=300)
        print("Plot successfully saved as 'loss_curve.png'.")
        
        # Display the plot
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found. Ensure it is in the same directory as this script.")

# Run the function (ensure your file is named 'training.log')
plot_loss_curve('/home/jay_agarwal_2022/kd-lgatr/mlp_scratch/training.log')