import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Constants
SCALE_FACTOR = 20.0

# ==========================================
# 1. THE BASELINE MLP ARCHITECTURE
# ==========================================
class MLPTagger(nn.Module):
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
# 2. RAW CARTESIAN DATASET
# ==========================================
class RawCartesianDataset(Dataset):
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
        raw_p4 = self.p4[idx] 
        
        # Truncate or Pad to 128 particles exactly
        if raw_p4.shape[0] >= self.max_particles:
            features = raw_p4[:self.max_particles, :]
        else:
            padding = torch.zeros((self.max_particles - raw_p4.shape[0], 4))
            features = torch.cat([raw_p4, padding], dim=0)
            
        # Crucial: Scale the Cartesian coordinates so the MLP can process them
        features = features / SCALE_FACTOR
            
        return features.view(-1), self.labels[idx]

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def main():
    # --- CONFIGURATIONS ---
    DATA_PATH = "/home/jay_agarwal_2022/lorentz-gatr/data/toptagging_full.npz" 
    EPOCHS = 100
    BATCH_SIZE = 256
    LR = 0.0002
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading (Parallelized)
    train_dataset = RawCartesianDataset(DATA_PATH, mode="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # Model Initialization
    model = MLPTagger(d_input=4, d_ff=512, d_output=1, depth=3, dropout=0.0)

    # GPU Parallelization
    if torch.cuda.device_count() > 1:
        print(f"Parallelizing across {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10)

    print("\n--- Starting Raw Cartesian Baseline Training ---")
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

    # Save logic that safely unwraps DataParallel
    save_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(save_model.state_dict(), "baseline_raw_cartesian_mlp.pt")
    print("\nTraining Complete! Raw Cartesian weights saved.")

if __name__ == "__main__":
    main()