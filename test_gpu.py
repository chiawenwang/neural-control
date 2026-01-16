# For testing MPS (Metal Performance Shaders) availability on macOS
import torch
print("torch:", torch.__version__)
print("mps available:", torch.backends.mps.is_available())
print("mps built:", torch.backends.mps.is_built())
if torch.backends.mps.is_available():
    x = torch.ones(1, device="mps")
    print("tensor device:", x.device)