import torch
from torch import nn
from .text_encoder import TextEncoder
from .image_encoder import ImageEncoder
from .simple_tokenizer import tokenize
from .context_embedding import ContextEmbedding
from .context_prompt import ContextPrompt

class CepCLIP(nn.Module):
    def __init__(self, args: str, device: str):
        super(CepCLIP, self).__init__()

        self.args = args
        self.device = device
        self.image_size = args.image_size

        pretrained_path = f'./pretrained_weights/{args.model}.pt'

        with open(pretrained_path, 'rb') as opened_file:
            model = torch.jit.load(opened_file, map_location='cpu').eval()
            state_dict = model.state_dict()

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

        self.text_encoder = TextEncoder(context_length=context_length,
                                        vocab_size=vocab_size,
                                        transformer_width=transformer_width,
                                        transformer_heads=transformer_heads,
                                        transformer_layers=transformer_layers,
                                        embed_dim=embed_dim)

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        vision_heads = vision_width // 64
        self.image_encoder = ImageEncoder(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            num_tokens=args.num_tokens,
            prompt_dim=vision_width
          )

        self.context_prompt = ContextPrompt(input_dim=vision_width,
                                            prompt_dim=embed_dim, prompt_len=args.text_prompt_len)

        self.context_embedding = ContextEmbedding(dim_v=vision_width, dim_t=embed_dim,
                                                  dim_out=embed_dim, num_heads=args.num_heads)

        self.init_weights(checkpoint=state_dict)
        self.image_encoder.eval()
        self.text_encoder.eval()

        self.authentic_text_captions = ["a photo of a real face"]
        self.forged_text_captions = ["a photo of a forged face"]
        self.g_project = nn.Linear(vision_width, embed_dim*2)

    def init_weights(self, checkpoint):

        text_state_dict = {}
        image_state_dict = {}

        for k in checkpoint.keys():
            if k.startswith('transformer.'):
                text_state_dict[k] = checkpoint[k]

            elif k == 'positional_embedding' or k == 'text_projection' or k.startswith(
                    'token_embedding') or k.startswith('ln_final'):
                if k == 'positional_embedding' and checkpoint[k].size(0) > self.text_encoder.context_length:
                    checkpoint[k] = checkpoint[k][:self.text_encoder.context_length]
                    print('positional_embedding is tuncated from 77 to', self.text_encoder.context_length)
                text_state_dict[k] = checkpoint[k]

            elif k.startswith('visual.'):
                new_k = k.replace('visual.', '')
                image_state_dict[new_k] = checkpoint[k]

        u, w = self.text_encoder.load_state_dict(text_state_dict, False)
        print(u, w, 'are misaligned params in text encoder')

        u, w = self.image_encoder.load_state_dict(image_state_dict, False)
        print(u, w, 'are misaligned params in image encoder')

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    def forward(self, image):

        global_features, visual_embedding, _ = self.image_encoder(image)

        context_prompts = self.context_prompt(global_features)
        authentic_sentences = tokenize(self.authentic_text_captions).to(self.device)
        forged_sentences = tokenize(self.forged_text_captions).to(self.device)
        authentic_embeddings = self.text_encoder(authentic_sentences, context_prompts, self.dtype)
        forged_embeddings = self.text_encoder(forged_sentences, context_prompts, self.dtype)
        text_embeddings = torch.cat([authentic_embeddings, forged_embeddings], dim=1)

        text_embeddings = self.context_embedding(text_embeddings, visual_embedding)

        global_features = self.g_project(global_features)
        global_features = global_features / global_features.norm(dim=1, keepdim=True)
        text_embeddings = text_embeddings.permute(0, 2, 1)

        forgery_score = global_features.unsqueeze(1) @ text_embeddings
        forgery_score = forgery_score.squeeze(1)
        forgery_score = torch.softmax(forgery_score, dim=1)
        return forgery_score
