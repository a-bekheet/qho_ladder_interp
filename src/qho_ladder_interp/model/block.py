from tokenize import Single
import torch
import torch.nn as nn

from .attention import SingleHeadSelfAttention

class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()

        self.attn = SingleHeadSelfAttention(d_model=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_x = self.attn(x)

        x = x + delta_x

        return x