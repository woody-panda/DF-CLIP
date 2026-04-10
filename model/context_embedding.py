import torch
from torch import Tensor, nn
from typing import Tuple, Type
import numpy as np
from torch.nn import functional as F


class ContextEmbedding(nn.Module):

    def __init__(self, dim_v, dim_t, dim_out, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0., qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.dim_out = dim_out

        self.scale = qk_scale or dim_out ** -0.5

        self.q_proj = nn.Conv1d(dim_t, dim_out, kernel_size=1)
        self.k_proj = nn.Conv1d(dim_v, dim_out, kernel_size=1)
        self.v_proj = nn.Conv1d(dim_v, dim_out, kernel_size=1)
        self.proj_post = nn.Conv1d(dim_out, dim_out, kernel_size=1)

        self.beta = 1
        self.alpha = 0.25
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, E, I):


        B2, N2, C2 = I.shape

        B1, N1, C1 = E.shape

        Q = self.q_proj(E.permute(0, 2, 1)).permute(0, 2, 1).reshape(B1, N1, self.num_heads, self.head_dim)  # 1
        K = self.k_proj(I.permute(0, 2, 1)).permute(0, 2, 1).reshape(B2, N2, self.num_heads, self.head_dim)  # 1
        V = self.v_proj(I.permute(0, 2, 1)).permute(0, 2, 1).reshape(B2, N2, self.num_heads, self.head_dim)  # 1

        attn = torch.einsum('bnkc,bmkc->bknm', Q, K) * self.beta
        attn = attn.softmax(dim=-1)
        U = torch.einsum('bknm,bmkc->bnkc', attn, V).reshape(B1, N1, self.dim_out)
        U = self.proj_post(U.permute(0, 2, 1)).permute(0, 2, 1)
        U = U / U.norm(dim=-1, keepdim=True)

        H = torch.cat([E, U], dim=2)

        return H
