import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# lgatr package imports
from lgatr import LGATr, MLPConfig, SelfAttentionConfig
from lgatr.interface.vector import embed_vector
from lgatr.interface.scalar import embed_scalar

# Constants
EPS = 1e-5
SCALE_FACTOR = 20.0

# ==========================================
# 1. TEACHER MODEL (L-GATr Wrapper)
# ==========================================
def embed_point(p4):
    """Embeds 4-momenta (E, px, py, pz) into a Geometric Algebra Multivector."""
    # The new lgatr natively takes the full 4D Lorentz vector
    mv = embed_vector(p4)
    
    # Add the required channel dimension -> [B, N, 1, 16]
    return mv.unsqueeze(-2)

class TeacherLGATr(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        
        self.encoder = LGATr(
            in_mv_channels=1,
            out_mv_channels=1,     # Outputs 1 multivector channel
            hidden_mv_channels=16,
            in_s_channels=1,       
            out_s_channels=None,   # FIX 1: No scalar projection layer!
            hidden_s_channels=32,
            num_blocks=12,
            attention=SelfAttentionConfig(
                num_heads=8,
                dropout_prob=0.0,
                increase_hidden_channels=2
            ),
            mlp=MLPConfig(
                dropout_prob=0.0,
                increase_hidden_channels=2
            ) 
        )

        if checkpoint_path:
            self._load_weights(checkpoint_path)

    def _load_weights(self, path):
        print(f"Loading Teacher weights from {path}...")
        
        import lgatr
        if 'gatr' not in sys.modules:
            sys.modules['gatr'] = lgatr
            
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)))
        
        clean_state_dict = {k[4:] if k.startswith("net.") else k: v for k, v in state_dict.items()}

        missing, unexpected = self.encoder.load_state_dict(clean_state_dict, strict=False)
        print(f"✓ Loaded L-GATr. Missing: {len(missing)} | Unexpected: {len(unexpected)}")

    def forward(self, p4, mask):
        batch_size = p4.shape[0]
        device = p4.device

        # Inject Beam Token [1, 0, 0, 1] at index 0
        beam_p4 = torch.tensor([1.0, 0.0, 0.0, 1.0], device=device).view(1, 1, 4).expand(batch_size, -1, -1)
        p4_in = torch.cat([beam_p4, p4], dim=1)        
        
        beam_mask = torch.ones((batch_size, 1), device=device, dtype=mask.dtype)
        mask_in = torch.cat([beam_mask, mask], dim=1) 
        mask_in = mask_in.unsqueeze(1).unsqueeze(2)

        # Embedding
        x_mv = embed_point(p4_in)          
        
        # ==========================================
        # FIX: The True L-GATr Scalar Feature!
        # It expects 1.0 for the beam token, and 0.0 for all other particles.
        # ==========================================
        x_s = torch.zeros((batch_size, p4_in.shape[1], 1), device=device, dtype=p4_in.dtype)
        x_s[:, 0, 0] = 1.0  

        # Forward pass
        out_mv, out_s = self.encoder(x_mv, x_s, attn_mask=mask_in)
        
        # Readout from the Beam Token
        logits = out_mv[:, 0, 0, 0] 
        
        return logits


# ==========================================
# 2. STUDENT MODEL (MLP)
# ==========================================
class StudentMLP(nn.Module):
    """Standard 3-layer MLP Student matching the old config."""
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
# 3. DENSE DATASET & COLLATE FUNCTION
# ==========================================
class KDDataset(Dataset):
    def __init__(self, filename, mode, dtype=torch.float32):
        super().__init__()
        data = np.load(filename)
        self.kinematics = torch.tensor(data[f"kinematics_{mode}"], dtype=dtype)
        self.labels = torch.tensor(data[f"labels_{mode}"], dtype=torch.bool)
        self.num_jets, self.num_constituents, _ = self.kinematics.shape

    def __len__(self):
        return self.num_jets

    def __getitem__(self, idx):
        raw_p4 = self.kinematics[idx]
        label = self.labels[idx]

        # Student: Flattened, scaled vector
        mlp_input = (raw_p4 / SCALE_FACTOR).view(-1)

        # Teacher: Filter out zero-padding, keep valid particles, scaled
        mask = (raw_p4.abs() > EPS).all(dim=-1)
        teacher_p4 = raw_p4[mask] / SCALE_FACTOR

        return mlp_input, teacher_p4, label

def kd_collate_fn(batch):
    mlp_inputs, teacher_p4_list, labels = [], [], []

    for mlp_in, t_p4, lbl in batch:
        mlp_inputs.append(mlp_in)
        teacher_p4_list.append(t_p4)
        labels.append(lbl)

    # Standard batching for Student
    batched_mlp = torch.stack(mlp_inputs, dim=0)
    batched_labels = torch.stack(labels, dim=0)

    # Dense padding for Teacher (Creates [Batch, max_N, 4])
    batched_teacher_p4 = pad_sequence(teacher_p4_list, batch_first=True, padding_value=0.0)
    
    # Create Attention Mask (True for real particles, False for padding zeros)
    teacher_mask = (batched_teacher_p4[..., 0] != 0.0)

    return batched_mlp, batched_teacher_p4, teacher_mask, batched_labels


# ==========================================
# 4. TRAINING LOOP
# ==========================================
def main():
    # --- HYPERPARAMETERS ---
    DATA_PATH = "/home/jay_agarwal_2022/lorentz-gatr/data/toptagging_full.npz"
    TEACHER_CHECKPOINT = "/home/jay_agarwal_2022/lorentz-gatr/runs/topt/GATr_7327/models/model_run0_it169999.pt"
    
    BATCH_SIZE = 128
    LR = 0.001
    ALPHA = 0.5         # lambda from your config
    TEMPERATURE = 4.0   # T from your config
    EPOCHS = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(f"Using device: {device}")

    # 1. Dataset
    dataset = KDDataset(DATA_PATH, mode="train")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=kd_collate_fn)

    # 2. Initialize Teacher 
    teacher_model = TeacherLGATr(checkpoint_path=TEACHER_CHECKPOINT).to(device)
    teacher_model.eval()

    # 3. Initialize Student
    student_model = StudentMLP(
        in_features=dataset.num_constituents * 4, 
        d_ff=512, 
        dropout=0.2
    ).to(device)
    
    optimizer = optim.Adam(student_model.parameters(), lr=LR)

    # 4. Loop
    print("\n--- Starting Knowledge Distillation ---")
    for epoch in range(EPOCHS):
        student_model.train()
        total_loss = 0.0
        
        for batch_idx, (mlp_inputs, teacher_p4, teacher_mask, labels) in enumerate(dataloader):
            mlp_inputs = mlp_inputs.to(device)
            teacher_p4 = teacher_p4.to(device)
            teacher_mask = teacher_mask.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()

            # Teacher Logits
            with torch.no_grad():
                teacher_logits = teacher_model(teacher_p4, teacher_mask)

            # Student Logits
            student_logits = student_model(mlp_inputs)

            # KD Loss
            soft_loss = F.binary_cross_entropy_with_logits(
                student_logits / TEMPERATURE,
                torch.sigmoid(teacher_logits / TEMPERATURE)
            ) * (TEMPERATURE ** 2)
            
            hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels)
            loss = (ALPHA * soft_loss) + ((1.0 - ALPHA) * hard_loss)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    torch.save(student_model.state_dict(), "distilled_student_mlp.pt")
    print("\nTraining Complete! Student saved.")

if __name__ == "__main__":
    main()