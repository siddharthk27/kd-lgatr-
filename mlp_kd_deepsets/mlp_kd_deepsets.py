"""
DeepSets / PFN-style student distilled from L-GATr top-tagging teacher.

v1 design (see readme.md for full rationale):
  - Teacher: lloca-experiments LGATrWrapper, loaded from
    seed1001/model_run0_it174999.pt (in_s_channels=8, 7 tagging features +
    3 spurions + global-token flag). We import the wrapper instead of
    re-implementing — prior KD scripts (in_s_channels=1, single beam token)
    were silently mismatched and produced garbage teacher signal.
  - Student: per-particle MLP (7 → 128 → 128) → masked sum+mean+max pool
    (→ 384) → head MLP (384 → 256 → 128 → 1). LayerNorm, GELU, ~150k params.
  - Features: same 7 tagging features the teacher consumes
    (log_pt, log_E, log_pt_rel, log_E_rel, dphi, deta, dr), pre-normalized
    via the same TAGGING_FEATURES_PREPROCESSING table.
  - KD loss: alpha · MSE(student_logit/T, teacher_logit/T) · T^2
              + (1 - alpha) · BCE(student_logit, label)
    with T=2, alpha=0.7. Logit-MSE (not BCE-on-soft-probs) since this is a
    binary classifier with a confident teacher.
  - Teacher logits are pre-computed once over train+val and cached to disk
    so the per-epoch step is student-only.
  - Tracking: val AUC every epoch, best-val checkpoint saved.
"""
import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

# ---- import teacher pipeline from Jay's repo ----
LLOCA_REPO = "/nfs_home/users/dhruvk/scratch/jay_agarwal/lgatr/lloca-experiments"
JAY_REPO   = "/nfs_home/users/dhruvk/jay_agarwal/lgatr/repos/lloca-experiments"
for p in (LLOCA_REPO, JAY_REPO):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from experiments.tagging.wrappers import LGATrWrapper           # noqa: E402
from experiments.tagging.embedding import embed_tagging_data    # noqa: E402
from lloca.framesnet.nonequi_frames import IdentityFrames       # noqa: E402
from lgatr import LGATr, MLPConfig, SelfAttentionConfig         # noqa: E402

# ============================================================
# Constants
# ============================================================
DATA_PATH    = "/nfs_home/users/dhruvk/scratch/jay_agarwal/lgatr/data/toptagging/toptagging_full.npz"
TEACHER_CKPT = "/nfs_home/users/dhruvk/scratch/jay_agarwal/lgatr/lloca-experiments/runs/topt_lgatr/seed1001/models/model_run0_it174999.pt"

# tagging feature standardization (same table as the teacher's training)
# rows: (mean, factor)  ->  feature := (raw - mean) * factor
TAGGING_NORM = torch.tensor([
    [1.7,  0.7],   # log_pt
    [2.0,  0.7],   # log_energy
    [-4.7, 0.7],   # log_pt_rel
    [-4.7, 0.7],   # log_energy_rel
    [0.0,  1.0],   # dphi
    [0.0,  1.0],   # deta
    [0.2,  4.0],   # dr
], dtype=torch.float32)
N_FEAT = 7
EPS = 1e-5


def build_student_inputs(kin, mask, with_pairwise=False):
    """Concatenate the 7 base features with the 4 pairwise aggregates if requested."""
    base = compute_student_features(kin, mask, norm_table=TAGGING_NORM)        # (B, N, 7)
    if with_pairwise:
        pair = compute_pairwise_features(kin, mask)                            # (B, N, 4)
        return torch.cat([base, pair], dim=-1)                                 # (B, N, 11)
    return base


def student_in_dim(with_pairwise=False):
    return N_FEAT + (N_PAIR_FEAT if with_pairwise else 0)

# ============================================================
# cfg_data — must match teacher's training-time data config
# (from runs/topt_lgatr/seed1001/config_0.yaml, "data:" block)
# ============================================================
def make_cfg_data():
    return SimpleNamespace(
        tagging_features="all",
        boost_jet=False,
        beam_reference="lightlike",
        two_beams=True,
        add_time_reference=True,
        mass_reg=0.005,
        spurion_scale=1,
        max_particles=None,
    )

# ============================================================
# Dataset: load dense kinematics, return (kinematics, label, idx)
# ============================================================
class TopTaggingNPZ(Dataset):
    def __init__(self, path, mode):
        super().__init__()
        d = np.load(path)
        self.kin = torch.from_numpy(np.asarray(d[f"kinematics_{mode}"], dtype=np.float32))
        self.lbl = torch.from_numpy(np.asarray(d[f"labels_{mode}"], dtype=np.float32))
        self.mode = mode
        print(f"[data:{mode}] kin={tuple(self.kin.shape)} lbl={tuple(self.lbl.shape)} "
              f"top_frac={self.lbl.float().mean().item():.3f}")

    def __len__(self):
        return self.kin.shape[0]

    def __getitem__(self, idx):
        return self.kin[idx], self.lbl[idx], idx

# ============================================================
# Student-side engineered features (the same 7 the teacher consumes).
# Computed batched on GPU in the train loop, not in __getitem__,
# to avoid 8-worker CPU contention on a 24-CPU node.
# ============================================================
def compute_student_features(kin_dense, mask_dense, eps=1e-10, norm_table=None):
    """
    kin_dense  : (B, N, 4)   E, px, py, pz   (zero-padded)
    mask_dense : (B, N)      True where particle is real
    Returns
        features : (B, N, 7)  zero on padded rows
    """
    E  = kin_dense[..., 0]
    px = kin_dense[..., 1]
    py = kin_dense[..., 2]
    pz = kin_dense[..., 3]
    pt = torch.sqrt(px * px + py * py + eps)
    p  = torch.sqrt(px * px + py * py + pz * pz + eps)

    # jet 4-momentum (sum over real particles)
    jet = (kin_dense * mask_dense.unsqueeze(-1)).sum(dim=1, keepdim=True)         # (B,1,4)
    jE  = jet[..., 0]
    jpx = jet[..., 1]
    jpy = jet[..., 2]
    jpz = jet[..., 3]
    jpt = torch.sqrt(jpx * jpx + jpy * jpy + eps)
    jp  = torch.sqrt(jpx * jpx + jpy * jpy + jpz * jpz + eps)

    log_pt          = pt.clamp(min=eps).log()
    log_E           = E.clamp(min=eps).log()
    log_pt_rel      = log_pt - jpt.clamp(min=eps).log()
    log_E_rel       = log_E  - jE.clamp(min=eps).log()
    phi             = torch.atan2(py, px)
    phi_jet         = torch.atan2(jpy, jpx)
    dphi            = ((phi - phi_jet + math.pi) % (2 * math.pi)) - math.pi
    eta             = torch.atanh((pz / p).clamp(-0.999, 0.999))
    eta_jet         = torch.atanh((jpz / jp).clamp(-0.999, 0.999))
    deta            = -(eta - eta_jet)              # sign matches teacher
    dr              = torch.sqrt((dphi ** 2 + deta ** 2).clamp(min=eps))

    feats = torch.stack([log_pt, log_E, log_pt_rel, log_E_rel, dphi, deta, dr], dim=-1)
    if norm_table is not None:
        mean   = norm_table[:, 0].to(feats.device, feats.dtype)
        factor = norm_table[:, 1].to(feats.device, feats.dtype)
        feats  = (feats - mean) * factor
    feats = feats * mask_dense.unsqueeze(-1)
    return feats


# ============================================================
# Pairwise interaction features for the DeepSets student.
#
# DeepSets has no two-particle interaction structure built in. We give it
# four per-particle aggregates that summarize each particle's relationship
# with the rest of the jet — informative for top vs QCD substructure.
# Non-equivariant, but plain scalars so the student can use them naturally.
#
# For each particle i (over valid neighbors j ≠ i in the same jet):
#   1. dR_min               nearest-neighbor distance in (η, φ)
#   2. log_kT_min           log of the smallest pairwise k_T = min(pT_i, pT_j) · ΔR_ij
#   3. n_close_03           count of valid j with ΔR_ij < 0.3
#   4. dR_to_hardest        ΔR from i to the highest-pT particle in the jet
# ============================================================
N_PAIR_FEAT = 4

def compute_pairwise_features(kin_dense, mask_dense, eps=1e-10):
    """
    kin_dense  : (B, N, 4)   E, px, py, pz   (zero-padded)
    mask_dense : (B, N) bool True where particle is real
    Returns: (B, N, N_PAIR_FEAT), zero on padded rows
    """
    B, N, _ = kin_dense.shape
    device  = kin_dense.device
    dtype   = kin_dense.dtype
    BIG     = torch.tensor(1.0e9, device=device, dtype=dtype)

    px = kin_dense[..., 1]
    py = kin_dense[..., 2]
    pz = kin_dense[..., 3]
    pt = torch.sqrt(px * px + py * py + eps)
    p  = torch.sqrt(px * px + py * py + pz * pz + eps)
    eta = torch.atanh((pz / p).clamp(-0.999, 0.999))
    phi = torch.atan2(py, px)

    pair_mask = mask_dense.unsqueeze(2) & mask_dense.unsqueeze(1)             # (B, N, N)
    diag = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
    pair_mask = pair_mask & ~diag                                              # exclude self
    has_neighbor = pair_mask.any(dim=2)                                        # (B, N)

    deta = eta.unsqueeze(2) - eta.unsqueeze(1)                                # (B, N, N)
    dphi = phi.unsqueeze(2) - phi.unsqueeze(1)
    dphi = ((dphi + math.pi) % (2 * math.pi)) - math.pi
    dR   = torch.sqrt(deta * deta + dphi * dphi + eps)                        # (B, N, N)

    # 1) nearest-neighbor distance
    dR_for_min = dR.masked_fill(~pair_mask, BIG)
    dR_min, _ = dR_for_min.min(dim=2)
    dR_min = torch.where(has_neighbor, dR_min, torch.zeros_like(dR_min))

    # 2) min log k_T over neighbors
    pt_pair_min = torch.minimum(pt.unsqueeze(2), pt.unsqueeze(1))             # (B, N, N)
    kT          = pt_pair_min * dR
    kT_for_min  = kT.masked_fill(~pair_mask, BIG)
    kT_min, _   = kT_for_min.min(dim=2)
    kT_min      = torch.where(has_neighbor, kT_min.clamp(min=eps),
                              torch.full_like(kT_min, eps))
    log_kT_min  = kT_min.log()

    # 3) neighbor count within ΔR < 0.3
    n_close_03 = ((dR < 0.3) & pair_mask).sum(dim=2).to(dtype)

    # 4) ΔR to hardest particle in the jet
    pt_for_max = pt.masked_fill(~mask_dense, -1.0)
    hardest_idx = pt_for_max.argmax(dim=1)                                    # (B,)
    # pull dR[b, i, hardest_idx[b]] for every (b, i)
    hardest_idx_exp = hardest_idx.view(B, 1, 1).expand(-1, N, 1)              # (B, N, 1)
    dR_to_hardest = dR.gather(2, hardest_idx_exp).squeeze(-1)                 # (B, N)
    particle_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
    is_self_hardest = (particle_idx == hardest_idx.unsqueeze(1))
    dR_to_hardest = dR_to_hardest.masked_fill(is_self_hardest | ~mask_dense, 0.0)

    feats = torch.stack([dR_min, log_kT_min, n_close_03, dR_to_hardest], dim=-1)  # (B, N, 4)
    feats = feats * mask_dense.unsqueeze(-1)
    return feats


# ============================================================
# Teacher-side: dense -> sparse (fourmomenta, scalars=empty, ptr)
# Matches lloca-experiments dense_to_sparse_jet but works on (B,N,4).
# ============================================================
def dense_to_sparse(kin_dense, mask_dense):
    fm     = kin_dense[mask_dense]                                  # (n_total, 4)
    counts = mask_dense.sum(dim=-1)                                 # (B,)
    ptr    = torch.zeros(counts.numel() + 1, dtype=torch.long, device=kin_dense.device)
    ptr[1:] = torch.cumsum(counts, dim=0)
    scalars = torch.zeros(fm.shape[0], 0, dtype=fm.dtype, device=fm.device)
    return fm, scalars, ptr

# ============================================================
# Student: PFN-style DeepSets
# ============================================================
class DeepSetsStudent(nn.Module):
    def __init__(self, in_dim=7, phi_hidden=128, rho_hidden=(256, 128), p_drop=0.1):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, phi_hidden),
            nn.LayerNorm(phi_hidden),
            nn.GELU(),
            nn.Linear(phi_hidden, phi_hidden),
            nn.LayerNorm(phi_hidden),
            nn.GELU(),
            nn.Linear(phi_hidden, phi_hidden),
        )
        rho_layers = []
        d = phi_hidden * 3   # sum + mean + max
        for h in rho_hidden:
            rho_layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p_drop)]
            d = h
        rho_layers.append(nn.Linear(d, 1))
        self.rho = nn.Sequential(*rho_layers)

    def pool(self, feats, mask):
        """Returns (z, h) where z is the (B, 3H) pooled jet vector."""
        h = self.phi(feats)                          # (B, N, H)
        m = mask.unsqueeze(-1).to(h.dtype)           # (B, N, 1)

        h_masked = h * m
        s_sum    = h_masked.sum(dim=1)                                # (B, H)
        counts   = m.sum(dim=1).clamp(min=1.0)                        # (B, 1)
        s_mean   = s_sum / counts
        h_for_max = h.masked_fill(~mask.unsqueeze(-1), torch.finfo(h.dtype).min)
        s_max     = h_for_max.max(dim=1).values                       # (B, H)
        return torch.cat([s_sum, s_mean, s_max], dim=-1)              # (B, 3H)

    def forward(self, feats, mask, return_pooled=False):
        """
        feats : (B, N, in_dim)
        mask  : (B, N) bool
        Returns:
            logits           (B,)
            pooled (optional)(B, 3H) — request with return_pooled=True
        """
        z = self.pool(feats, mask)
        logits = self.rho(z).squeeze(-1)
        if return_pooled:
            return logits, z
        return logits


class HintProjector(nn.Module):
    """Small head mapping the student's pooled jet vector to the teacher's
    penultimate-invariant target (48 dims by default)."""
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return self.net(z)

# ============================================================
# Teacher loader (architecture matches seed1001/config_0.yaml exactly)
# ============================================================
def build_teacher(checkpoint_path, attention_backend, device):
    net = LGATr(
        in_mv_channels=1,
        out_mv_channels=1,
        hidden_mv_channels=16,
        in_s_channels=8,
        out_s_channels=None,
        hidden_s_channels=32,
        num_blocks=12,
        attention=SelfAttentionConfig(
            num_heads=8,
            multi_query=False,
            increase_hidden_channels=2,
            head_scale=True,
        ),
        mlp=MLPConfig(
            activation="gelu",
            increase_hidden_channels=2,
        ),
    )

    # The wrapper takes a partial-net factory in Hydra; mimic that here.
    def net_factory(out_mv_channels):
        # out_mv_channels is set by the wrapper to out_channels=1; net is already built
        # with out_mv_channels=1, so just return it.
        return net

    wrapper = LGATrWrapper(
        net=net_factory,
        framesnet=IdentityFrames(),
        out_channels=1,
        mean_aggregation=False,
        use_amp=False,
        attention_backend=attention_backend,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    missing, unexpected = wrapper.load_state_dict(state_dict, strict=False)
    print(f"[teacher] loaded from {os.path.basename(checkpoint_path)} | "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 50 or len(unexpected) > 50:
        print(f"[teacher] WARN: large missing/unexpected counts — interface may be wrong")
        print(f"  first missing: {missing[:5]}")
        print(f"  first unexpected: {unexpected[:5]}")

    wrapper.to(device).eval()
    for p in wrapper.parameters():
        p.requires_grad = False
    return wrapper

# ============================================================
# Build the embedding dict and run teacher on a batched sparse jet
# ============================================================
@torch.no_grad()
def teacher_logits(wrapper, fm_sparse, scalars_sparse, ptr, cfg_data):
    # embed_tagging_data mutates ptr in-place; clone it
    embedding = embed_tagging_data(
        fourmomenta=fm_sparse.clone(),
        scalars=scalars_sparse.clone(),
        ptr=ptr.clone(),
        cfg_data=cfg_data,
    )
    out = wrapper(embedding)
    # wrapper returns (logits, {}, None) — see lloca-experiments wrappers.py
    logits = out[0] if isinstance(out, (tuple, list)) else out
    return logits.squeeze(-1)


# ============================================================
# Hint distillation: extract penultimate hidden state at the global token.
#
# Architecture sanity:
#   LGATr = linear_in -> [block_0, ..., block_{N-1}] -> linear_out
# The state right after blocks[-1] is the deepest representation; linear_out
# collapses it to (out_mv_channels=1, 16-dim mv). For our teacher,
#   block output multivectors: shape (1, total_tokens, hidden_mv=16, 16)
#   block output scalars:      shape (1, total_tokens, hidden_s=32)
# At the global token we extract:
#   - scalar component (idx 0) of each MV channel       -> 16 numbers
#   - all scalar-stream channels                        -> 32 numbers
#   = 48 plain scalars per jet (no equivariant features
#     the non-equivariant student would have to memorize in lab frame)
# ============================================================
class HookedTeacher:
    """
    Wraps the LGATr teacher to expose, after each forward, a per-jet
    48-dim "invariant" tensor extracted from the penultimate state at the
    global token. Forward returns (logits, invariants).
    """
    HIDDEN_MV = 16
    HIDDEN_S  = 32
    INVARIANT_DIM = HIDDEN_MV + HIDDEN_S   # 48

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self._penult_mv = None
        self._penult_s  = None
        # wrapper.net is the LGATr instance; hook the last block
        self._handle = wrapper.net.blocks[-1].register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        # LGATrBlock.forward returns (mv, s)
        mv, s = output
        # detach to avoid keeping graph during inference; we're under no_grad anyway
        self._penult_mv = mv.detach()
        self._penult_s  = s.detach()

    def remove_hook(self):
        self._handle.remove()

    @torch.no_grad()
    def forward(self, embedding):
        # `embed_tagging_data` has already added spurions and bumped ptr by n_spurions.
        # The wrapper.forward will further bump ptr by +1 per jet for the global token.
        # Pre-global-insertion ptr is what we need to compute global token positions.
        ptr_pre_global = embedding["ptr"].clone()
        batchsize = ptr_pre_global.numel() - 1
        global_idxs = ptr_pre_global[:-1] + torch.arange(
            batchsize, device=ptr_pre_global.device
        )

        out = self.wrapper(embedding)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits = logits.squeeze(-1)

        # mv shape: (1, total_tokens, hidden_mv=16, 16);  s shape: (1, total_tokens, 32)
        mv_at_g = self._penult_mv[0, global_idxs]      # (B, 16, 16)
        s_at_g  = self._penult_s[0,  global_idxs]      # (B, 32)
        mv_scalars = mv_at_g[..., 0]                   # (B, 16) — scalar component per MV channel
        invariants = torch.cat([mv_scalars, s_at_g], dim=-1)   # (B, 48)
        return logits, invariants


@torch.no_grad()
def teacher_logits_and_invariants(hooked, fm_sparse, scalars_sparse, ptr, cfg_data):
    embedding = embed_tagging_data(
        fourmomenta=fm_sparse.clone(),
        scalars=scalars_sparse.clone(),
        ptr=ptr.clone(),
        cfg_data=cfg_data,
    )
    return hooked.forward(embedding)

# ============================================================
# Pre-compute teacher logits (and optionally penultimate invariants) over a split
# ============================================================
def precompute_teacher_logits(split_name, dataset, wrapper, cfg_data,
                              cache_path, device, batch_size=256, num_workers=4,
                              hooked=None, invariant_cache_path=None):
    """
    Pre-compute and cache teacher logits.

    If hooked (a HookedTeacher) and invariant_cache_path are both supplied,
    also cache penultimate invariants in the same forward pass. The two caches
    are kept separate so the existing logits cache stays valid across versions.
    """
    cache = Path(cache_path)
    inv_cache = Path(invariant_cache_path) if invariant_cache_path else None
    want_invariants = hooked is not None and inv_cache is not None

    # Short-circuit: everything we need is already on disk.
    if cache.exists() and (not want_invariants or inv_cache.exists()):
        print(f"[cache] loading {split_name} teacher logits from {cache}")
        logits_t = torch.load(cache, map_location="cpu")
        if want_invariants:
            print(f"[cache] loading {split_name} teacher invariants from {inv_cache}")
            inv_t = torch.load(inv_cache, map_location="cpu")
            return logits_t, inv_t
        return logits_t

    # Logits already cached but invariants are not — preserve the existing logit
    # cache (don't overwrite v1's frozen file) and only fill in invariants.
    have_logits = cache.exists()
    if have_logits and want_invariants:
        print(f"[cache] logits already at {cache}; computing only invariants")
        cached_logits = torch.load(cache, map_location="cpu")
    else:
        cached_logits = None

    label = "logits + invariants" if want_invariants else "logits"
    if have_logits and want_invariants:
        label = "invariants (existing logits preserved)"
    print(f"[cache] computing {split_name} teacher {label} over {len(dataset)} jets ...")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=_collate_for_teacher_only,
    )

    out = torch.empty(len(dataset), dtype=torch.float32)
    inv_out = (torch.empty(len(dataset), HookedTeacher.INVARIANT_DIM, dtype=torch.float32)
               if want_invariants else None)
    # Sanity tracker for fresh-vs-cached logit drift (when both exist)
    max_logit_drift = 0.0
    pos = 0
    t0 = time.time()
    for kin, _lbl, _idx in loader:
        kin  = kin.to(device, non_blocking=True)
        mask = (kin.abs() > EPS).all(dim=-1)
        fm, sc, ptr = dense_to_sparse(kin, mask)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            if want_invariants:
                logits, invariants = teacher_logits_and_invariants(
                    hooked, fm, sc, ptr, cfg_data
                )
            else:
                logits = teacher_logits(wrapper, fm, sc, ptr, cfg_data)
                invariants = None
        n = logits.numel()
        fresh = logits.detach().to("cpu", torch.float32)
        out[pos:pos + n] = fresh
        if cached_logits is not None:
            drift = (cached_logits[pos:pos + n] - fresh).abs().max().item()
            if drift > max_logit_drift:
                max_logit_drift = drift
        if want_invariants:
            inv_out[pos:pos + n] = invariants.detach().to("cpu", torch.float32)
        pos += n
        if pos % (batch_size * 100) < batch_size:
            print(f"  [cache:{split_name}] {pos}/{len(dataset)}  "
                  f"elapsed={time.time()-t0:.1f}s")
    print(f"[cache:{split_name}] done in {time.time()-t0:.1f}s")
    if cached_logits is not None:
        print(f"[cache:{split_name}] max |fresh - cached| logit drift = {max_logit_drift:.2e}")

    # Preserve the original logits cache if it already existed (don't clobber v1).
    if not have_logits:
        torch.save(out, cache)
    if want_invariants:
        torch.save(inv_out, inv_cache)
        # Return the cached logits if we had them, else the fresh ones.
        return (cached_logits if cached_logits is not None else out), inv_out
    return out

def _collate_for_teacher_only(batch):
    kin = torch.stack([b[0] for b in batch], dim=0)
    lbl = torch.stack([b[1] for b in batch], dim=0)
    idx = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return kin, lbl, idx

# ============================================================
# KD loss (with optional penultimate-invariant hint term)
# ============================================================
def kd_loss(student_logits, teacher_logits, labels, alpha, T,
            student_hint=None, teacher_hint=None, beta=0.0,
            loss_type='logit_mse'):
    """
    L = α · soft  +  (1−α) · BCE(s, label)  [+ β · MSE(proj(s_pool), t_invariants)]

    loss_type:
      'logit_mse'   — soft = MSE(s/T, t/T) · T²    (T cancels exactly; F2)
      'softmax_kl'  — soft = KL(p_t || p_s) · T², where p_· = softmax([0, z]/T)
                      (canonical Hinton-style; T is a real lever here)
    """
    if loss_type == 'logit_mse':
        soft = F.mse_loss(student_logits / T, teacher_logits.detach() / T) * (T * T)
    elif loss_type == 'softmax_kl':
        s2 = torch.stack([torch.zeros_like(student_logits), student_logits], dim=-1) / T
        t2 = torch.stack([torch.zeros_like(teacher_logits), teacher_logits.detach()], dim=-1) / T
        log_p_s = F.log_softmax(s2, dim=-1)
        p_t     = F.softmax(t2, dim=-1)
        soft = F.kl_div(log_p_s, p_t, reduction='batchmean') * (T * T)
    else:
        raise ValueError(f"Unknown kd loss_type: {loss_type!r}")
    hard = F.binary_cross_entropy_with_logits(student_logits, labels.float())
    total = alpha * soft + (1.0 - alpha) * hard

    if beta > 0.0 and student_hint is not None and teacher_hint is not None:
        hint = F.mse_loss(student_hint, teacher_hint.detach())
        total = total + beta * hint
        return total, soft.detach(), hard.detach(), hint.detach()
    return total, soft.detach(), hard.detach(), None

# ============================================================
# Eval helpers
# ============================================================
@torch.no_grad()
def eval_split(student, dataset, device, batch_size=1024, num_workers=4,
               with_pairwise=False):
    student.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=_collate_for_teacher_only,
    )
    all_lbl, all_pred = [], []
    for kin, lbl, _idx in loader:
        kin = kin.to(device, non_blocking=True)
        mask = (kin.abs() > EPS).all(dim=-1)
        feats = build_student_inputs(kin, mask, with_pairwise=with_pairwise)
        logits = student(feats, mask)
        all_lbl.append(lbl.numpy())
        all_pred.append(torch.sigmoid(logits).float().cpu().numpy())
    y_true = np.concatenate(all_lbl)
    y_pred = np.concatenate(all_pred)
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    rej = {}
    for eps_S in (0.3, 0.5, 0.8):
        i = int(np.argmin(np.abs(tpr - eps_S)))
        rej[eps_S] = float("inf") if fpr[i] == 0 else 1.0 / float(fpr[i])
    return auc, rej, y_true, y_pred

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch-size",     type=int,   default=512)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--weight-decay",   type=float, default=1e-4)
    parser.add_argument("--alpha",          type=float, default=0.7)
    parser.add_argument("--temperature",    type=float, default=2.0)
    parser.add_argument("--num-workers",    type=int,   default=8)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--out-dir",        type=str,   default=".")
    parser.add_argument("--teacher-cache-dir", type=str, default=None,
                        help="Where teacher_logits_{train,val}.pt live. Defaults to --out-dir. "
                             "Point ablation runs at the v1 cache to skip teacher entirely.")
    parser.add_argument("--smoke-test",     action="store_true",
                        help="Just verify teacher loads + reproduces AUC on a tiny val subset, then exit.")
    parser.add_argument("--attention-backend", type=str, default="xformers",
                        help="LGATr attention backend (lloca wrapper supports: xformers | flash | flex). "
                             "xformers matches the teacher's training-time setting.")
    parser.add_argument("--hint-beta",      type=float, default=0.0,
                        help="Phase 3: weight on penultimate-invariant hint loss. "
                             "0 disables the hint (Phase 1/2 behavior). Try 0.1–1.0.")
    parser.add_argument("--hint-projector-hidden", type=int, default=128,
                        help="Hidden width of the student's hint-projection head.")
    parser.add_argument("--use-pairwise", action="store_true",
                        help="Add 4 pairwise-aggregate features per particle "
                             "(dR_min, log_kT_min, n_close_03, dR_to_hardest). "
                             "Brings student input from 7 to 11 dims; "
                             "teacher cache is unchanged so v1 cache is reused.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.teacher_cache_dir) if args.teacher_cache_dir else out_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = cache_dir / "teacher_logits_train.pt"
    val_cache   = cache_dir / "teacher_logits_val.pt"
    train_inv_cache = cache_dir / "teacher_invariants_train.pt"
    val_inv_cache   = cache_dir / "teacher_invariants_val.pt"
    cache_complete = train_cache.exists() and val_cache.exists()
    hint_mode = args.hint_beta > 0.0
    inv_cache_complete = train_inv_cache.exists() and val_inv_cache.exists()

    # determinism
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} | torch={torch.__version__} | seed={args.seed}")
    if device.type == "cuda":
        print(f"[setup] gpu={torch.cuda.get_device_name(0)} | "
              f"vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    cfg_data = make_cfg_data()
    print(f"[setup] cfg_data={vars(cfg_data)}")
    print(f"[setup] teacher cache dir={cache_dir} | cache_complete={cache_complete} | "
          f"hint_mode={hint_mode} | inv_cache_complete={inv_cache_complete}")

    # ---- decide whether we need the teacher this run ----
    # We can skip teacher entirely iff:
    #   - logit cache exists (always required), AND
    #   - if hint_mode, the invariant cache also exists.
    val_ds = TopTaggingNPZ(DATA_PATH, mode="val")
    train_teacher_invariants = None
    val_teacher_invariants   = None
    can_skip_teacher = (cache_complete
                        and (not hint_mode or inv_cache_complete)
                        and not args.smoke_test)

    if can_skip_teacher:
        print("[teacher] cache present — skipping teacher load + smoke test")
        train_ds = TopTaggingNPZ(DATA_PATH, mode="train")
        train_teacher_logits = torch.load(train_cache, map_location="cpu")
        val_teacher_logits   = torch.load(val_cache,   map_location="cpu")
        if train_teacher_logits.numel() != len(train_ds):
            raise RuntimeError(
                f"train cache size {train_teacher_logits.numel()} "
                f"!= dataset size {len(train_ds)}; cache is stale, delete it"
            )
        if val_teacher_logits.numel() != len(val_ds):
            raise RuntimeError(
                f"val cache size {val_teacher_logits.numel()} "
                f"!= dataset size {len(val_ds)}; cache is stale, delete it"
            )
        if hint_mode:
            train_teacher_invariants = torch.load(train_inv_cache, map_location="cpu")
            val_teacher_invariants   = torch.load(val_inv_cache,   map_location="cpu")
            if train_teacher_invariants.shape != (len(train_ds), HookedTeacher.INVARIANT_DIM):
                raise RuntimeError(
                    f"train invariant cache shape {train_teacher_invariants.shape} "
                    f"!= ({len(train_ds)}, {HookedTeacher.INVARIANT_DIM}); stale"
                )
            if val_teacher_invariants.shape != (len(val_ds), HookedTeacher.INVARIANT_DIM):
                raise RuntimeError(
                    f"val invariant cache shape {val_teacher_invariants.shape} "
                    f"!= ({len(val_ds)}, {HookedTeacher.INVARIANT_DIM}); stale"
                )
    else:
        teacher = build_teacher(TEACHER_CKPT, args.attention_backend, device)
        hooked = HookedTeacher(teacher) if hint_mode else None

        # ---- smoke test: teacher AUC on a small val slice ----
        smoke_n = min(2048, len(val_ds))
        smoke_loader = DataLoader(
            torch.utils.data.Subset(val_ds, list(range(smoke_n))),
            batch_size=256, shuffle=False, num_workers=2,
            collate_fn=_collate_for_teacher_only,
        )
        print(f"[smoke] running teacher over {smoke_n} val jets ...")
        t_logits, t_labels = [], []
        for kin, lbl, _idx in smoke_loader:
            kin = kin.to(device, non_blocking=True)
            mask = (kin.abs() > EPS).all(dim=-1)
            fm, sc, ptr = dense_to_sparse(kin, mask)
            with torch.cuda.amp.autocast(dtype=torch.float32):
                tl = teacher_logits(teacher, fm, sc, ptr, cfg_data)
            t_logits.append(tl.float().cpu().numpy())
            t_labels.append(lbl.numpy())
        t_logits = np.concatenate(t_logits)
        t_labels = np.concatenate(t_labels)
        smoke_auc = roc_auc_score(t_labels, t_logits)
        print(f"[smoke] teacher AUC on {smoke_n} val jets = {smoke_auc:.4f}  "
              f"(expected ~0.987; if << 0.95 the teacher interface is wrong)")
        if args.smoke_test:
            return
        if smoke_auc < 0.95:
            raise RuntimeError(
                f"Teacher AUC {smoke_auc:.4f} too low — likely architecture/spurion/feature "
                f"mismatch with the checkpoint. Aborting before wasting compute on KD."
            )

        # ---- pre-compute teacher logits (and optionally invariants) ----
        train_ds = TopTaggingNPZ(DATA_PATH, mode="train")
        train_out = precompute_teacher_logits(
            "train", train_ds, teacher, cfg_data,
            cache_path=train_cache,
            device=device, batch_size=256, num_workers=args.num_workers,
            hooked=hooked, invariant_cache_path=(train_inv_cache if hint_mode else None),
        )
        val_out = precompute_teacher_logits(
            "val", val_ds, teacher, cfg_data,
            cache_path=val_cache,
            device=device, batch_size=256, num_workers=args.num_workers,
            hooked=hooked, invariant_cache_path=(val_inv_cache if hint_mode else None),
        )
        if hint_mode:
            train_teacher_logits, train_teacher_invariants = train_out
            val_teacher_logits,   val_teacher_invariants   = val_out
        else:
            train_teacher_logits = train_out
            val_teacher_logits   = val_out

        if hooked is not None:
            hooked.remove_hook()
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()

    val_auc_teacher = roc_auc_score(val_ds.lbl.numpy(), val_teacher_logits.numpy())
    print(f"[teacher] full val AUC (from cache) = {val_auc_teacher:.4f}")

    # ---- student ----
    phi_hidden = 128
    in_dim = student_in_dim(with_pairwise=args.use_pairwise)
    student = DeepSetsStudent(in_dim=in_dim, phi_hidden=phi_hidden,
                              rho_hidden=(256, 128), p_drop=0.1).to(device)
    pooled_dim = phi_hidden * 3   # sum + mean + max
    hint_proj = (HintProjector(pooled_dim, HookedTeacher.INVARIANT_DIM,
                               hidden=args.hint_projector_hidden).to(device)
                 if hint_mode else None)
    print(f"[student] in_dim={in_dim} (use_pairwise={args.use_pairwise})")

    n_params_s = sum(p.numel() for p in student.parameters())
    n_params_h = sum(p.numel() for p in hint_proj.parameters()) if hint_proj else 0
    print(f"[student] DeepSets PFN | params={n_params_s:,}"
          + (f" + hint_proj={n_params_h:,}" if hint_proj else ""))

    train_params = list(student.parameters())
    if hint_proj is not None:
        train_params += list(hint_proj.parameters())
    optimizer = torch.optim.AdamW(train_params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_ds) / args.batch_size) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler()

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=_collate_for_teacher_only, drop_last=False,
    )

    history = []
    best_val_auc = -1.0
    best_path = out_dir / "deepsets_student_best.pt"

    print(f"\n--- Training: {args.epochs} epochs, BS {args.batch_size}, "
          f"alpha={args.alpha}, T={args.temperature}"
          + (f", hint_beta={args.hint_beta}" if hint_mode else "")
          + " ---\n")
    for epoch in range(1, args.epochs + 1):
        student.train()
        if hint_proj is not None:
            hint_proj.train()
        t0 = time.time()
        running = {"loss": 0.0, "soft": 0.0, "hard": 0.0, "hint": 0.0, "n": 0}
        for kin, lbl, idx in train_loader:
            kin = kin.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            t_log = train_teacher_logits[idx].to(device, non_blocking=True)
            t_inv = (train_teacher_invariants[idx].to(device, non_blocking=True)
                     if hint_mode else None)
            mask = (kin.abs() > EPS).all(dim=-1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                feats = build_student_inputs(kin, mask, with_pairwise=args.use_pairwise)
                if hint_mode:
                    s_log, s_pool = student(feats, mask, return_pooled=True)
                    s_hint = hint_proj(s_pool)
                else:
                    s_log = student(feats, mask)
                    s_hint = None
                loss, soft, hard, hint = kd_loss(
                    s_log, t_log, lbl,
                    alpha=args.alpha, T=args.temperature,
                    student_hint=s_hint, teacher_hint=t_inv, beta=args.hint_beta,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = lbl.numel()
            running["loss"] += loss.item() * bs
            running["soft"] += soft.item() * bs
            running["hard"] += hard.item() * bs
            if hint is not None:
                running["hint"] += hint.item() * bs
            running["n"]    += bs

        avg_loss = running["loss"] / running["n"]
        avg_soft = running["soft"] / running["n"]
        avg_hard = running["hard"] / running["n"]
        avg_hint = running["hint"] / running["n"] if hint_mode else 0.0

        # ---- validation ----
        val_auc, val_rej, _, _ = eval_split(student, val_ds, device,
                                            batch_size=1024, num_workers=args.num_workers,
                                            with_pairwise=args.use_pairwise)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        components = f"soft={avg_soft:.4f} hard={avg_hard:.4f}"
        if hint_mode:
            components += f" hint={avg_hint:.4f}"
        print(f"epoch {epoch:3d}/{args.epochs} | "
              f"loss={avg_loss:.4f} ({components}) | "
              f"val AUC={val_auc:.4f} rej@0.3={val_rej[0.3]:.0f} rej@0.5={val_rej[0.5]:.0f} | "
              f"lr={lr_now:.2e} | {elapsed:.1f}s")

        hist_entry = dict(epoch=epoch, loss=avg_loss, soft=avg_soft, hard=avg_hard,
                          val_auc=val_auc, val_rej_30=val_rej[0.3],
                          val_rej_50=val_rej[0.5], val_rej_80=val_rej[0.8],
                          lr=lr_now, time_s=elapsed)
        if hint_mode:
            hist_entry["hint"] = avg_hint
        history.append(hist_entry)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            ckpt = {
                "model_state_dict": student.state_dict(),
                "epoch": epoch,
                "val_auc": val_auc,
                "val_rej": val_rej,
                "args": vars(args),
            }
            if hint_proj is not None:
                ckpt["hint_proj_state_dict"] = hint_proj.state_dict()
            torch.save(ckpt, best_path)
            print(f"  -> new best val AUC, saved to {best_path}")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ---- final test eval using best-val checkpoint ----
    print(f"\n[final] loading best-val checkpoint (val AUC={best_val_auc:.4f}) ...")
    state = torch.load(best_path, map_location=device)
    student.load_state_dict(state["model_state_dict"])
    test_ds = TopTaggingNPZ(DATA_PATH, mode="test")
    test_auc, test_rej, _, _ = eval_split(student, test_ds, device,
                                          batch_size=1024, num_workers=args.num_workers,
                                          with_pairwise=args.use_pairwise)
    print(f"[final] test AUC={test_auc:.4f} | "
          f"rej@0.3={test_rej[0.3]:.0f}  rej@0.5={test_rej[0.5]:.0f}  rej@0.8={test_rej[0.8]:.0f}")

    with open(out_dir / "final_test_metrics.json", "w") as f:
        json.dump({"test_auc": test_auc,
                   "test_rej": {str(k): v for k, v in test_rej.items()},
                   "best_val_auc": best_val_auc}, f, indent=2)

if __name__ == "__main__":
    main()
