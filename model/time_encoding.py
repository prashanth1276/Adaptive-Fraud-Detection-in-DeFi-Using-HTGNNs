"""
Temporal positional encoding module for the Heterogeneous Temporal Graph Neural Network.

This module implements a sinusoidal time encoding scheme derived from the Time2Vec
formulation, adapted for relative temporal distances in DeFi transaction graphs.
It operates at the edge level, converting scalar delta-t values (elapsed time in days)
into dense continuous embeddings. These embeddings are subsequently aggregated via
message passing and concatenated with structural node features before heterogeneous
graph convolution. This module is active during both training and inference.
"""

import math

import torch
import torch.nn as nn


class TimeEncoding(nn.Module):
    """Sinusoidal relative time encoder for temporal edge embeddings.

    Encodes scalar elapsed-time values (delta_t in days) into fixed-dimensional
    embeddings using alternating sine and cosine projections at geometrically
    spaced frequencies. This encoding is isomorphic to the positional encoding
    used in Transformer architectures (Vaswani et al., 2017) but applied to
    continuous temporal distances rather than discrete sequence positions.

    The frequency spectrum is log-uniformly distributed over [1/10000, 1],
    enabling the model to capture both short-term (intra-day) and long-term
    (multi-month) temporal dependencies simultaneously.

    Attributes:
        time_dim (int): Dimensionality of the output time embedding vector.
        freq (torch.Tensor): Registered buffer of shape (time_dim // 2,)
            containing the pre-computed angular frequencies.
    """

    def __init__(self, time_dim: int):
        """Initializes the TimeEncoding module with pre-computed frequency buffers.

        Args:
            time_dim (int): Output embedding dimensionality. Must be even to
                accommodate paired sine/cosine projections. Typical values
                are 16, 32, or 64.

        Raises:
            ValueError: If time_dim is odd, as sine/cosine pairing requires
                an equal split of the embedding dimension.
        """
        super().__init__()

        # Require even dimension to allow symmetric sin/cos decomposition
        if time_dim % 2 != 0:
            raise ValueError("time_dim must be even for sin/cos encoding")

        self.time_dim = time_dim

        # Compute log-uniformly spaced frequencies: exp(-k * log(10000) / D)
        # for k in {0, 2, 4, ..., D-2}. This spans low-frequency (slow trends)
        # to high-frequency (rapid fluctuation) temporal patterns.
        freq = torch.exp(
            torch.arange(0, time_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / time_dim)
        )
        self.register_buffer("freq", freq)

    def forward(self, delta_t):
        """Encodes a batch of temporal distances into sinusoidal embedding vectors.

        Args:
            delta_t (torch.Tensor): 1-D tensor of shape (E,) containing elapsed
                time values in seconds (converted from Unix timestamps). Values
                are expected to be non-negative after temporal edge masking.

        Returns:
            torch.Tensor: Embedding tensor of shape (E, time_dim), where each
                row is a concatenation of sin and cos projections at all
                frequencies, capturing multi-scale temporal context.
        """
        # Ensure dtype and device alignment with the registered frequency buffer
        delta_t = delta_t.float().to(self.freq.device)

        # Deprecated alternative implementation retained for historical reference.
        # delta_t = torch.abs(delta_t)

        # Expand delta_t to (E, 1) for broadcasting against freq shape (D/2,)
        delta_t = delta_t.unsqueeze(-1)

        # Compute outer product: angles[i, k] = delta_t[i] * freq[k]
        angles = delta_t * self.freq

        # Concatenate sin and cos along the feature axis to produce (E, time_dim)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb
