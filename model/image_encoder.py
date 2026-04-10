import torch
import torch.nn.functional as F
from torch.nn import Dropout
from torch import nn, autocast

from functools import reduce
from operator import mul

import math
from .transformer import LayerNorm, Transformer


class ImageEncoder(nn.Module):
    def __init__(self, input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=512,
                 drop_path_rate=0.0, out_indices=[3, 5, 7, 11], get_embeddings=True,
                 num_tokens=20, prompt_dim=512, **kwargs):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.spatial_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings
        self.num_layers = layers

        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)

        self.out_indices = out_indices

        if get_embeddings:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        embed_dim = width

        ## Setting of visual prompt tuning
        self.num_tokens = num_tokens
        self.prompt_dim = prompt_dim

        ## Add the prompt parameters # exclude_key=prompt:
        self._init_prompt(patch_size, self.num_tokens, self.prompt_dim)

    def _init_prompt(self, patch, num_tokens, prompt_dim):
        patch_size = []
        patch_size.append(patch)
        patch_size.append(patch)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

        self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
        nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
        self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
        self.prompt_dropout = Dropout(0.1)



    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            if 'positional_embedding' in state_dict.keys():
                if self.positional_embedding.shape != state_dict['positional_embedding'].shape:
                    # (1025, 768)                      (197, 768)
                    print(
                        f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}')
                    cls_pos = state_dict["positional_embedding"][0:1, :]

                    spatial_pos = F.interpolate(
                        state_dict["positional_embedding"][1:, ].reshape(1, 14, 14, 768).permute(0, 3, 1, 2),
                        size=(self.spatial_size, self.spatial_size), mode='bilinear')
                    spatial_pos = spatial_pos.reshape(768, self.spatial_size * self.spatial_size).permute(1, 0)
                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                    state_dict['positional_embedding'] = positional_embedding
                    assert self.positional_embedding.shape == state_dict['positional_embedding'].shape

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in vision transformer')

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0, :] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:, ].reshape(1, self.spatial_size, self.spatial_size, C).permute(0, 3, 1, 2),
                                    size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos
        x = self.ln_pre(x)

        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
            x[:, 1:, :]
        ), dim=1)

        x = x.permute(1, 0, 2)

        features = []
        outs = []
        
        for i, blk in enumerate(self.transformer.resblocks):
            x = blk(x)
            if len(self.out_indices) > 1:
                if i in self.out_indices:
                    xp = x.permute(1, 0, 2)[:, 1 + self.num_tokens:, :].permute(0, 2, 1).reshape(B, -1, H, W)
                    features.append(xp.contiguous())

        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        # x = x @ self.proj

        global_embedding = x[:, 0]
        visual_embedding = x[:, 1+ self.num_tokens:]
        # visual_embedding = visual_embedding / visual_embedding.norm(dim=1, keepdim=True)

        return global_embedding, visual_embedding, features

