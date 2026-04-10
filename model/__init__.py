from .cep_clip import *
from .text_encoder import *
from .transformer import *


def load_model(args, device):
    model = CepCLIP(args=args, device=device).to(device)

    return model