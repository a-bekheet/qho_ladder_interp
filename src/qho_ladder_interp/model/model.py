import torch
import torch.nn as nn

from qho_ladder_interp.tokens import VOCAB
from .embeddings import TokenPositionalEmbedding
from .block import ResidualAttentionBlock

D_MODEL = 128
T_MAX = 8

class MinimalTransformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.embed = TokenPositionalEmbedding() # maps token IDs + positons -> stream vectors

        self.block = ResidualAttentionBlock(d_model=D_MODEL) # single resattenionblock 

        self.unembed = nn.Linear(D_MODEL, len(VOCAB), bias=False) # maps stream -> logits over vocab

    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids) # b,t,d
        x = self.block(x) #b,t,d
        logits = self.unembed(x) # b,t,|V|

        return logits