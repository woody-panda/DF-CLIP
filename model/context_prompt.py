import torch
import torch.nn as nn
import numpy as np


class ContextPrompt(nn.Module):
    def __init__(self, input_dim, prompt_dim, prompt_len, k=2):
        super().__init__()
        self.prompt_query = nn.Parameter(torch.randn(1, prompt_len, prompt_dim))
        self.project = nn.Linear(input_dim, prompt_dim)
        nn.init.trunc_normal_(self.prompt_query)
        self._initialize_weights()

    #
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

    def forward(self, global_feat, use_global=True):
        B, C = global_feat.shape
        global_feat_new = self.project(global_feat.reshape(B, 1, C))
        prompt_query = self.prompt_query + torch.zeros((B, self.prompt_query.shape[-2], self.prompt_query.shape[-1]),
                                                       dtype=self.prompt_query.dtype, device=self.prompt_query.device)
        if use_global:
            class_feature = prompt_query + global_feat_new
        else:
            class_feature = prompt_query
        return class_feature
