import math
import torch
import torch.nn as nn

class SingleHeadSelfAttention(nn.Module):
    """
    implements Attention(X) = softmax (Q K^T / sqrt(d_k)) V

    with:
        Q = X W_Q
        K = X W_K
        V = X W_V

    shapes:
        X : (B,T,D)
        Q,K,V : (B,T,D)
        Attention Output: (B,T,D)
    """

    def __init__(self, d_model: int):
        super().__init__()

        self.d_model = d_model
        
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: tensor of shape (B,T,D)

        Returns
        _______
        tensor of shape (B,T,D)
        """
        B, T, D = x.shape
        assert D == self.d_model, "Input dimensionality mismatch"

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # compute raw attention scores
        # scores[b, i, j] = <Q[b,i], K[b,j]> / sqrt(D)
        scores = torch.matmul(Q, K.transpose(-1, 2))
        scores = scores / math.sqrt(self.d_model) # b,t,t

        attn_weights = torch.softmax(scores, dim=-1) #b,t,t

        out = torch.matmul(attn_weights, V) #b,t,d

        return out