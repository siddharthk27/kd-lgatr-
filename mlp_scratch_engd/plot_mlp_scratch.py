import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Constants
SCALE_FACTOR = 20.0

# ==========================================
# 1. STUDENT MODEL
# ==========================================
class StudentMLP(nn.Module):
    def __init__(self, in_features, d_ff=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ==========================================
# 2. EVALUATION DATASET
# ==========================================
class MLPTestDataset(Dataset):
    def __init__(self, filename, mode="test", dtype=torch.float32):
        super().__init__()
        data = np.load(filename)
        self.kinematics = torch.tensor(data[f"kinematics_{mode}"], dtype=dtype)
        self.labels = torch.tensor(data[f"labels_{mode}"], dtype=torch.float32)
        self.num_jets, self.num_constituents, self.num_features = self.kinematics.shape

    def __len__(self):
        return self.num_jets

    def __getitem__(self, idx):
        raw_p4 = self.kinematics[idx]
        mlp_input = (raw_p4 / SCALE_FACTOR).view(-1)
        return mlp_input, self.labels[idx]

# ==========================================
# 3. PLOTTING SCRIPT
# ==========================================
def main():
    DATA_PATH = "/home/jay_agarwal_2022/lorentz-gatr/data/toptagging_full.npz" 
    MODEL_WEIGHTS = "distilled_student_mlp.pt"     
    BATCH_SIZE = 1024

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating plots using device: {device}")

    # Load Data & Model
    dataset = MLPTestDataset(DATA_PATH, mode="test")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    in_features = dataset.num_constituents * 4 
    model = StudentMLP(in_features=in_features).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()

    # Collect Predictions
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            probs = torch.sigmoid(model(inputs.to(device)))
            all_labels.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    # Calculate ROC metrics
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # --- PLOT 1: ROC CURVE ---
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Distilled MLP (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Background Efficiency)', fontsize=12)
    plt.ylabel('True Positive Rate (Signal Efficiency)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) - Top Tagging', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved ROC Curve to 'roc_curve.png'")

    # --- PLOT 2: PROBABILITY DISTRIBUTION ---
    plt.figure(figsize=(8, 6))
    sig_preds = y_pred[y_true == 1]
    bkg_preds = y_pred[y_true == 0]

    # Plot histograms
    plt.hist(bkg_preds, bins=50, alpha=0.5, color='red', label='QCD Background', density=True)
    plt.hist(sig_preds, bins=50, alpha=0.5, color='blue', label='Top Signal', density=True)
    
    plt.xlabel('Network Output Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Student MLP Output Distribution', fontsize=14)
    plt.legend(loc='upper center', fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig('prob_dist.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved Probability Distribution to 'prob_dist.png'")

if __name__ == "__main__":
    main()
