import torch
import torch.nn as nn
import math

class TimeEncoding(nn.Module):
    """
    Relative time encoding (Δt in days)
    """
    def __init__(self, time_dim: int):
        super().__init__()

        # Ensure even dimension
        if time_dim % 2 != 0:
            raise ValueError("time_dim must be even for sin/cos encoding")

        self.time_dim = time_dim

        freq = torch.exp(
            torch.arange(0, time_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / time_dim)
        )
        self.register_buffer("freq", freq)

    def forward(self, delta_t):
        # delta_t: seconds → days
        delta_t = delta_t.float().to(self.freq.device)

        # Handle potential negative delta_t (if any timestamps are messy)
        #delta_t = torch.abs(delta_t)
        
        delta_t = delta_t.unsqueeze(-1)

        angles = delta_t * self.freq
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb
