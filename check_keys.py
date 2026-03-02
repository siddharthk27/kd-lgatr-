import torch
import sys

# Load model
from kd import TeacherLGATr
teacher = TeacherLGATr()
encoder_keys = set(teacher.encoder.state_dict().keys())

# Load checkpoint
path = "/home/jay_agarwal_2022/lorentz-gatr/runs/topt/GATr_7327/models/model_run0_it169999.pt"
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)))
clean_state_dict = {k[4:] if k.startswith("net.") else k: v for k, v in state_dict.items()}
ckpt_keys = set(clean_state_dict.keys())

# Compare
missing = encoder_keys - ckpt_keys
unexpected = ckpt_keys - encoder_keys

print(f"Missing keys ({len(missing)}):")
for k in sorted(list(missing)):
    print(f"  {k}")

print(f"\nUnexpected keys ({len(unexpected)}):")
for k in sorted(list(unexpected)):
    print(f"  {k}")
