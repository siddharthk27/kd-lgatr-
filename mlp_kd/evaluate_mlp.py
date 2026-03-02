import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

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
        print(f"Loading {mode} subset from {filename}...")
        
        # Load the unified .npz file
        data = np.load(filename)
        
        # Extract strictly the test arrays
        self.kinematics = torch.tensor(data[f"kinematics_{mode}"], dtype=dtype)
        self.labels = torch.tensor(data[f"labels_{mode}"], dtype=torch.float32)
        
        self.num_jets, self.num_constituents, self.num_features = self.kinematics.shape

    def __len__(self):
        return self.num_jets

    def __getitem__(self, idx):
        raw_p4 = self.kinematics[idx]
        label = self.labels[idx]

        # Pre-process exactly like training
        mlp_input = (raw_p4 / SCALE_FACTOR).view(-1)

        return mlp_input, label

# ==========================================
# 3. EVALUATION LOOP
# ==========================================
def main():
    # --- PATHS ---
    # Point this to the full npz file you used for training
    DATA_PATH = "/home/jay_agarwal_2022/lorentz-gatr/data/toptagging_full.npz" 
    MODEL_WEIGHTS = "distilled_student_mlp.pt"     
    BATCH_SIZE = 1024

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data (mode="test" ensures we only get the test split)
    dataset = MLPTestDataset(DATA_PATH, mode="test")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize Model & Load Weights
    in_features = dataset.num_constituents * 4 
    model = StudentMLP(in_features=in_features, d_ff=512, dropout=0.2).to(device)
    
    print(f"Loading student weights from {MODEL_WEIGHTS}...")
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    
    # CRITICAL: Disable dropout for evaluation
    model.eval()

    # 3. Collect Predictions
    all_labels = []
    all_preds = []

    print("Evaluating model...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    # ==========================================
    # 4. CALCULATE METRICS & OPTIMAL THRESHOLD
    # ==========================================
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Calculate Youden's J statistic to find the best threshold
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_thresh = thresholds[best_idx]
    
    # Recalculate Accuracy using the optimal threshold
    optimal_preds = (y_pred >= best_thresh).astype(int)
    best_acc = accuracy_score(y_true, optimal_preds)

    # Background Rejection (1 / fpr) at fixed Signal Efficiency (tpr)
    def get_rejection(target_tpr):
        idx = np.argmin(np.abs(tpr - target_tpr))
        if fpr[idx] == 0:
            return float('inf')
        return 1.0 / fpr[idx]

    rej_30 = get_rejection(0.3)
    rej_50 = get_rejection(0.5)
    rej_80 = get_rejection(0.8)

    # ==========================================
    # 5. PRINT RESULTS
    # ==========================================
    print("\n" + "="*45)
    print("        TEST SET EVALUATION RESULTS        ")
    print("="*45)
    print(f"AUC:                     {auc:.4f}")
    print(f"Optimal Threshold:       {best_thresh:.4f}")
    print(f"Calibrated Accuracy:     {best_acc:.4f}")
    print("-" + "-"*44)
    print("Background Rejection (1 / e_B):")
    print(f"  @ 30% Signal Eff:      {rej_30:.0f}")
    print(f"  @ 50% Signal Eff:      {rej_50:.0f}")
    print(f"  @ 80% Signal Eff:      {rej_80:.0f}")
    print("="*45)

if __name__ == "__main__":
    main()
