import torch
import torch.nn as nn

from qho_ladder_interp.tokens import VOCAB
from qho_ladder_interp.config import N_MAX

D_MODEL = 128
T_MAX = 8

class TokenPositionalEmbedding(nn.Module):
    """
    Maps token IDs and positions to vectors in R^{d_model}.
    """

    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding( # implements lookup table
            num_embeddings=len(VOCAB),
            embedding_dim=D_MODEL,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=T_MAX,
            embedding_dim=D_MODEL,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids: LongTensor of shape (B, T)

        Returns
        -------
        Tensor of shape (B, T, D_MODEL)
        """
        B, T = input_ids.shape

        if T > T_MAX:
            raise ValueError(f"Sequence length {T} exceeds T_MAX={T_MAX}")
        
        token_vecs = self.token_embedding(input_ids) # b,t,d
        positions = torch.arange(T, device=input_ids.device) #t,
        pos_vecs = self.position_embedding(positions) #t,c

        return token_vecs + pos_vecs