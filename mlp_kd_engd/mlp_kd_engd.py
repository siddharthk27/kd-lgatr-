import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import sys

# lgatr package imports
from lgatr import LGATr, MLPConfig, SelfAttentionConfig
from lgatr.interface.vector import embed_vector

# Constants
SCALE_FACTOR = 20.0

# ==========================================
# 1. TEACHER MODEL (L-GATr)
# ==========================================
def embed_point(p4):
    """Embeds 4-momenta (E, px, py, pz) into a Geometric Algebra Multivector."""
    mv = embed_vector(p4)
    return mv.unsqueeze(-2)

class TeacherLGATr(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        
        self.encoder = LGATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=16,
            in_s_channels=1,       
            out_s_channels=None,   # Pure multivector output
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

        x_mv = embed_point(p4_in)          
        
        # Scalar flag: 1.0 for the beam token, 0.0 for particles
        x_s = torch.zeros((batch_size, p4_in.shape[1], 1), device=device, dtype=p4_in.dtype)
        x_s[:, 0, 0] = 1.0  

        out_mv, out_s = self.encoder(x_mv, x_s, attn_mask=mask_in)
        
        # Readout: global token (seq 0), first channel (chan 0), scalar component (idx 0)
        logits = out_mv[:, 0, 0, 0] 
        return logits

# ==========================================
# 2. STUDENT MODEL (Engineered MLP)
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
# 3. ASYMMETRIC DATASET & COLLATE FUNCTION
# ==========================================
class AsymmetricKDDataset(Dataset):
    def __init__(self, filename, mode="train", max_particles=128):
        super().__init__()
        data = np.load(filename)
        self.kinematics = torch.tensor(data[f"kinematics_{mode}"], dtype=torch.float32)
        self.labels = torch.tensor(data[f"labels_{mode}"], dtype=torch.float32)
        self.max_particles = max_particles

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raw_p4 = self.kinematics[idx]
        label = self.labels[idx]

        # ----------------------------------------
        # A. TEACHER PREP (Raw & Scaled)
        # ----------------------------------------
        valid_mask_t = (raw_p4.abs() > 1e-5).all(dim=-1)
        teacher_p4 = raw_p4[valid_mask_t] / SCALE_FACTOR

        # ----------------------------------------
        # B. STUDENT PREP (Engineered Features)
        # ----------------------------------------
        mask_s = (raw_p4[:, 0] > 0).float()
        Pjet = (raw_p4 * mask_s.unsqueeze(-1)).sum(dim=0)
        
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
        features = features * mask_s.unsqueeze(-1)
        
        if features.shape[0] >= self.max_particles:
            features = features[:self.max_particles, :]
        else:
            padding = torch.zeros((self.max_particles - features.shape[0], 4))
            features = torch.cat([features, padding], dim=0)
            
        student_features = features.view(-1)

        return student_features, teacher_p4, label

def asymmetric_collate_fn(batch):
    student_inputs, teacher_p4_list, labels = [], [], []

    for s_in, t_p4, lbl in batch:
        student_inputs.append(s_in)
        teacher_p4_list.append(t_p4)
        labels.append(lbl)

    batched_student = torch.stack(student_inputs, dim=0)
    batched_labels = torch.stack(labels, dim=0)

    batched_teacher_p4 = pad_sequence(teacher_p4_list, batch_first=True, padding_value=0.0)
    teacher_mask = (batched_teacher_p4[..., 0] != 0.0)

    return batched_student, batched_teacher_p4, teacher_mask, batched_labels

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def main():
    # --- HYPERPARAMETERS ---
    DATA_PATH = "/home/jay_agarwal_2022/lorentz-gatr/data/toptagging_full.npz" 
    TEACHER_CHECKPOINT = "/home/jay_agarwal_2022/lorentz-gatr/runs/topt/GATr_7327/models/model_run0_it169999.pt"
    
    BATCH_SIZE = 512      
    LR = 0.0002          
    ALPHA = 0.5          
    TEMPERATURE = 4.0    
    EPOCHS = 100         

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading with CPU Parallelization
    dataset = AsymmetricKDDataset(DATA_PATH, mode="train")
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=asymmetric_collate_fn,
        num_workers=8,        
        pin_memory=True,      
        prefetch_factor=2     
    )

    # Teacher Setup
    teacher_model = TeacherLGATr(checkpoint_path=TEACHER_CHECKPOINT).to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Student Setup
    student_model = StudentMLP(d_input=4, d_ff=512, d_output=1, depth=6, dropout=0.0).to(device)
    
    # GPU Parallelization
    if torch.cuda.device_count() > 1:
        print(f"Parallelizing across {torch.cuda.device_count()} GPUs!")
        teacher_model = nn.DataParallel(teacher_model)
        student_model = nn.DataParallel(student_model)

    optimizer = optim.Adam(student_model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10)

    print("\n--- Starting Engineered Knowledge Distillation ---")
    for epoch in range(EPOCHS):
        student_model.train()
        total_loss = 0.0
        
        for batch_idx, (student_features, teacher_p4, teacher_mask, labels) in enumerate(dataloader):
            student_features = student_features.to(device)
            teacher_p4 = teacher_p4.to(device)
            teacher_mask = teacher_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher_model(teacher_p4, teacher_mask)

            student_logits = student_model(student_features)

            soft_loss = F.binary_cross_entropy_with_logits(
                student_logits / TEMPERATURE,
                torch.sigmoid(teacher_logits / TEMPERATURE)
            ) * (TEMPERATURE ** 2)
            
            hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels)
            
            loss = (ALPHA * soft_loss) + ((1.0 - ALPHA) * hard_loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 200 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(dataloader)} | Soft KD Loss: {soft_loss.item():.4f} | Hard Loss: {hard_loss.item():.4f} | Total Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"====> Epoch {epoch+1} Average KD Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Save logic that handles the DataParallel unwrapping automatically
    save_model = student_model.module if isinstance(student_model, nn.DataParallel) else student_model
    torch.save(save_model.state_dict(), "distilled_engineered_mlp_depth6.pt")
    print("\nTraining Complete! Engineered Student saved.")

if __name__ == "__main__":
    main()