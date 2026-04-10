import torch
from torch import nn
from .transformer import Transformer, LayerNorm
from torch import nn, autocast


class TextEncoder(nn.Module):
    def __init__(self, context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=1024):
        super().__init__()

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @autocast('cuda')
    def forward(self, text, context_prompt, dtype):
        # print(text)
        pos_x, pos_y = torch.where(text == 49407)

        x = self.token_embedding(text).type(dtype)
        N, L, D = x.shape

        text_feature_list = []
        for i in range(context_prompt.shape[0]):
            x_new = torch.zeros_like(x).to(x.device)
            for j in range(x.shape[0]):
                x_new[j, :, :] = torch.cat([x[j, 0:pos_y[j], :], context_prompt[i, :, :],
                                            x[j, (pos_y[j] + 1):(self.context_length - context_prompt.shape[1] + 1)]],
                                           dim=0).unsqueeze(0)

            x_new = x_new + self.positional_embedding.type(dtype)
            x_new = x_new.permute(1, 0, 2)
            x_new = self.transformer(x_new)

            x_new = x_new.permute(1, 0, 2)  # LND -> NLD
            x_new = self.ln_final(x_new).type(dtype)

            x_new = x_new[torch.arange(x_new.shape[0]), torch.where(text == 49407)[1] + context_prompt.shape[
                1] - 1] @ self.text_projection
            x_new = x_new / x_new.norm(dim=-1, keepdim=True)
            x_new = x_new.mean(dim=0, keepdim=True)
            x_new = x_new / x_new.norm(dim=-1, keepdim=True)
            text_feature_list.append(x_new)

        result = torch.stack(text_feature_list, dim=0)
        return result