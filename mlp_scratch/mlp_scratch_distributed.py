import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ==========================================
# 1. THE BASELINE MLP (Replicated)
# ==========================================
class MLPTagger(nn.Module):
    def __init__(self, d_input=4, d_ff=512, d_output=1, depth=3, dropout=0.0, max_particles=128):
        super().__init__()
        
        mlp = []
        d = d_input * max_particles
        
        # Loop matches the original codebase exactly
        for _ in range(depth - 1):
            mlp.extend([
                nn.Linear(d, d_ff, bias=True),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(d_ff),
            ])
            d = d_ff
        
        self.mlp = nn.Sequential(*mlp)
        
        # We use d_output=1 instead of 2 for standard Binary Cross Entropy
        self.output_layer = nn.Linear(d, d_output, bias=True)
        
    def forward(self, x):
        z = self.mlp(x)
        output = self.output_layer(z)
        return output.squeeze(-1)

# ==========================================
# 2. FEATURE ENGINEERING DATASET
# ==========================================
class BaselineDataset(Dataset):
    def __init__(self, filename, mode="train", max_particles=128):
        super().__init__()
        print(f"Loading {mode} data from {filename}...")
        data = np.load(filename)
        self.p4 = torch.tensor(data[f"kinematics_{mode}"], dtype=torch.float32)
        self.labels = torch.tensor(data[f"labels_{mode}"], dtype=torch.float32)
        self.max_particles = max_particles

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # [N, 4] -> E, px, py, pz
        p4 = self.p4[idx] 
        
        # Mask out zero-padded particles
        mask = (p4[:, 0] > 0).float()
        
        # Calculate Jet 4-momentum
        Pjet = (p4 * mask.unsqueeze(-1)).sum(dim=0)
        
        # --- FEATURE 1: Relative pT ---
        pt_part = torch.linalg.vector_norm(p4[:, 1:3], dim=-1)
        pt_jet = torch.linalg.vector_norm(Pjet[1:3], dim=-1)
        rel_pT = pt_part / (pt_jet + 1e-8)
        
        # --- FEATURE 2: Delta Eta (Pseudorapidity) ---
        p_mag_part = torch.linalg.vector_norm(p4[:, 1:4], dim=-1)
        p_mag_jet = torch.linalg.vector_norm(Pjet[1:4], dim=-1)
        
        # Clamp to avoid inf/nan at exactly 1.0 or -1.0
        eta_part = torch.atanh(torch.clamp(p4[:, 3] / (p_mag_part + 1e-8), -0.999, 0.999))
        eta_jet = torch.atanh(torch.clamp(Pjet[3] / (p_mag_jet + 1e-8), -0.999, 0.999))
        deta = eta_part - eta_jet
        
        # --- FEATURE 3: Delta Phi (Azimuthal Angle) ---
        phi_part = torch.atan2(p4[:, 2], p4[:, 1])
        phi_jet = torch.atan2(Pjet[2], Pjet[1])
        dphi = phi_part - phi_jet
        dphi = torch.remainder(dphi + torch.pi, 2 * torch.pi) - torch.pi
        
        # --- FEATURE 4: Relative Energy ---
        rel_E = p4[:, 0] / (Pjet[0] + 1e-8)
        
        # Combine features -> [N, 4]
        features = torch.stack([rel_pT, deta, dphi, rel_E], dim=-1)
        features = features * mask.unsqueeze(-1)
        
        # Truncate to 128 particles exactly as the original code did
        if features.shape[0] >= self.max_particles:
            features = features[:self.max_particles, :]
        else:
            # Pad if somehow smaller than 128 (though your data is likely 200)
            padding = torch.zeros((self.max_particles - features.shape[0], 4))
            features = torch.cat([features, padding], dim=0)
            
        return features.view(-1), self.labels[idx]

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def main():
    # --- CONFIGURATIONS ---
    DATA_PATH = "/home/jay_agarwal_2022/lorentz-gatr/data/toptagging_full.npz" 
    EPOCHS = 100
    
    # Scale dynamically based on available hardware
    NUM_GPUS = max(1, torch.cuda.device_count()) 
    BATCH_SIZE = 64 * NUM_GPUS
    LR = 0.0002 * NUM_GPUS
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} with {NUM_GPUS} GPUs")
    print(f"Global Batch Size: {BATCH_SIZE} | Learning Rate: {LR}")

    # Data
    train_dataset = BaselineDataset(DATA_PATH, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    # Model
    model = MLPTagger(d_input=4, d_ff=512, d_output=1, depth=3, dropout=0.0)
    
    # --- THE MULTI-GPU FIX ---
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    # Move the (now parallelized) model to the device
    model = model.to(device)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    # Optional: Scheduler mimicking the "warmup/patience/factor" config
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10)

    print("\n--- Starting Baseline Training ---")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 200 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(f"====> Epoch {epoch+1} Average Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    torch.save(model.state_dict(), "baseline_scratch_mlp.pt")
    print("\nTraining Complete! Baseline weights saved.")

if __name__ == "__main__":
    main()