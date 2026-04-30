import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

# ==========================================
# 1. STUDENT MODEL ARCHITECTURE
# ==========================================
class StudentMLP(nn.Module):
    def __init__(self, d_input=4, d_ff=512, d_output=1, depth=3, dropout=0.0, max_particles=128):
        super().__init__()
        
        mlp = []
        d = d_input * max_particles
        
        for _ in range(depth - 1):
            mlp.extend([
                nn.Linear(d, d_ff, bias=True),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(d_ff),
            ])
            d = d_ff
        
        self.mlp = nn.Sequential(*mlp)
        self.output_layer = nn.Linear(d, d_output, bias=True)
        
    def forward(self, x):
        z = self.mlp(x)
        output = self.output_layer(z)
        return output.squeeze(-1)

# ==========================================
# 2. FEATURE ENGINEERING DATASET
# ==========================================
class EngineeredStudentTestDataset(Dataset):
    def __init__(self, filename, mode="test", max_particles=128):
        super().__init__()
        print(f"Loading {mode} data from {filename}...")
        data = np.load(filename)
        self.kinematics = torch.tensor(data[f"kinematics_{mode}"], dtype=torch.float32)
        self.labels = torch.tensor(data[f"labels_{mode}"], dtype=torch.float32)
        self.max_particles = max_particles

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raw_p4 = self.kinematics[idx] 
        mask = (raw_p4[:, 0] > 0).float()
        Pjet = (raw_p4 * mask.unsqueeze(-1)).sum(dim=0)
        
        # --- Engineered Features (Must match KD training exactly) ---
        pt_part = torch.linalg.vector_norm(raw_p4[:, 1:3], dim=-1)
        pt_jet = torch.linalg.vector_norm(Pjet[1:3], dim=-1)
        rel_pT = pt_part / (pt_jet + 1e-8)
        
        p_mag_part = torch.linalg.vector_norm(raw_p4[:, 1:4], dim=-1)
        p_mag_jet = torch.linalg.vector_norm(Pjet[1:4], dim=-1)
        
        eta_part = torch.atanh(torch.clamp(raw_p4[:, 3] / (p_mag_part + 1e-8), -0.999, 0.999))
        eta_jet = torch.atanh(torch.clamp(Pjet[3] / (p_mag_jet + 1e-8), -0.999, 0.999))
        deta = eta_part - eta_jet
        
        phi_part = torch.atan2(raw_p4[:, 2], raw_p4[:, 1])
        phi_jet = torch.atan2(Pjet[2], Pjet[1])
        dphi = phi_part - phi_jet
        dphi = torch.remainder(dphi + torch.pi, 2 * torch.pi) - torch.pi
        
        rel_E = raw_p4[:, 0] / (Pjet[0] + 1e-8)
        
        features = torch.stack([rel_pT, deta, dphi, rel_E], dim=-1)
        features = features * mask.unsqueeze(-1)
        
        # Truncate or Pad to 128 particles
        if features.shape[0] >= self.max_particles:
            features = features[:self.max_particles, :]
        else:
            padding = torch.zeros((self.max_particles - features.shape[0], 4))
            features = torch.cat([features, padding], dim=0)
            
        return features.view(-1), self.labels[idx]

# ==========================================
# 3. EVALUATION & PLOTTING LOOP
# ==========================================
def main():
    # --- PATHS ---
    DATA_PATH = "/home/jay_agarwal_2022/lorentz-gatr/data/toptagging_full.npz" 
    MODEL_WEIGHTS = "distilled_engineered_mlp_depth6.pt" # Points to the newly trained KD model
    BATCH_SIZE = 1024

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data with multiprocessing for fast feature computation
    dataset = EngineeredStudentTestDataset(DATA_PATH, mode="test")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # 2. Initialize Model
    model = StudentMLP(d_input=4, d_ff=512, d_output=1, depth=6).to(device)
    
    # 3. Handle DataParallel Weights loading
    print(f"Loading weights from {MODEL_WEIGHTS}...")
    state_dict = torch.load(MODEL_WEIGHTS, map_location=device)
    clean_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    model.eval()

    # 4. Inference
    all_labels, all_preds = [], []
    print("Evaluating model...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            probs = torch.sigmoid(model(inputs))
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    # ==========================================
    # CALCULATE METRICS
    # ==========================================
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Youden's J statistic for optimal threshold
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_thresh = thresholds[best_idx]
    best_acc = accuracy_score(y_true, (y_pred >= best_thresh).astype(int))

    # Background Rejection (1 / fpr)
    def get_rejection(target_tpr):
        idx = np.argmin(np.abs(tpr - target_tpr))
        return float('inf') if fpr[idx] == 0 else 1.0 / fpr[idx]

    rej_30 = get_rejection(0.3)
    rej_50 = get_rejection(0.5)
    rej_80 = get_rejection(0.8)

    # ==========================================
    # PRINT RESULTS
    # ==========================================
    print("\n" + "="*50)
    print(" KD ENGINEERED STUDENT (MLP) TEST RESULTS ")
    print("="*50)
    print(f"AUC:                     {auc:.4f}")
    print(f"Optimal Threshold:       {best_thresh:.4f}")
    print(f"Calibrated Accuracy:     {best_acc:.4f}")
    print("-" + "-"*49)
    print("Background Rejection (1 / e_B):")
    print(f"  @ 30% Signal Eff:      {rej_30:.0f}")
    print(f"  @ 50% Signal Eff:      {rej_50:.0f}")
    print(f"  @ 80% Signal Eff:      {rej_80:.0f}")
    print("="*50)

    # ==========================================
    # GENERATE PLOTS
    # ==========================================
    # Plot 1: ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'KD Engineered MLP (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Background Efficiency)', fontsize=12)
    plt.ylabel('True Positive Rate (Signal Efficiency)', fontsize=12)
    plt.title('ROC Curve - KD Engineered Student', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve_kd_engineered_depth6.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved ROC Curve to 'roc_curve_kd_engineered.png'")

    # Plot 2: Probability Distribution
    plt.figure(figsize=(8, 6))
    sig_preds = y_pred[y_true == 1]
    bkg_preds = y_pred[y_true == 0]

    plt.hist(bkg_preds, bins=50, alpha=0.5, color='red', label='QCD Background', density=True)
    plt.hist(sig_preds, bins=50, alpha=0.5, color='blue', label='Top Signal', density=True)
    
    plt.axvline(best_thresh, color='black', linestyle='dashed', linewidth=2, label=f'Threshold ({best_thresh:.2f})')
    
    plt.xlabel('Network Output Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('KD Engineered Student Output Distribution', fontsize=14)
    plt.legend(loc='upper center', fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig('prob_dist_kd_engineered_depth6.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved Probability Distribution to 'prob_dist_kd_engineered.png'")

if __name__ == "__main__":
    main()