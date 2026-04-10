import torch
import argparse
from trainer import Trainer

def get_args():
    parser = argparse.ArgumentParser("DF-CLIP", add_help=True)
    # path
    parser.add_argument("--dataset_root", type=str, default='./datasets', help="model configs")
    parser.add_argument("--scenario", type=str, default='cross_dataset', help="scenario")
    parser.add_argument("--training_sets", type=str, nargs="+", default=['FF++'], help="training datasets")
    parser.add_argument("--test_sets", type=str, nargs="+", default=['FFIW', 'Celeb-DF', 'DFDC-P'],
                        help="test datasets")
    parser.add_argument("--learning_rate", type=float, default=0.00004, help="learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers")
    parser.add_argument("--log_steps", type=int, default=20, help="logging step")

    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    parser.add_argument("--out_indices", type=int, nargs="+",
                        default=[24], help="features used in image and text encoder")
    parser.add_argument("--num_tokens", type=int, default=20,
                        help="the length of the deep prompt tuning")
    parser.add_argument("--use_global", default=True, action="store_false",
                        help="Whether to use global visual features in the CEP module")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="head number in the CEE module")

    parser.add_argument("--text_prompt_len", type=int, default=2,
                        help="head number in the CEE module")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    device = 'cuda:0'
    args = get_args()
    trainer = Trainer(args=args, device=device)
    #with torch.no_grad():
    trainer.train()