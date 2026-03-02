import re
import matplotlib.pyplot as plt

def plot_kd_loss_curve(log_file_path):
    steps = []
    losses = []
    step_counter = 0

    # Regex pattern to match the new batch-level loss lines
    # Target: "Epoch 1 | Batch 100/9461 | Loss: 4.8520"
    pattern = re.compile(r'Epoch\s+\d+\s+\|\s+Batch\s+\d+/\d+\s+\|\s+Loss:\s+([\d.]+)')

    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    # Extract just the loss value
                    loss = float(match.group(1))
                    
                    steps.append(step_counter)
                    losses.append(loss)
                    # Increment step counter (each logged line represents a step in the plot)
                    step_counter += 1
                    
        if not steps:
            print("No training data found. Please check the log file format.")
            return

        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Using no marker since batch logs can get very dense
        plt.plot(steps, losses, marker='', linestyle='-', color='r', linewidth=1.5, label='Batch Loss')
        
        # Formatting the plot
        plt.title('Knowledge Distillation Training Loss', fontsize=14)
        plt.xlabel('Logged Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig('kd_loss_curve.png', dpi=300)
        print("Plot successfully saved as 'kd_loss_curve.png'.")
        
        # Display the plot
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found. Ensure it is in the same directory.")

# Run the function (ensure your file is named 'training.log')
plot_kd_loss_curve('training.log')