import argparse
import os
import torch
from data_utils import data_loader
from model import *
from model import CepCLIP
import torch
from metrics import dec_metrics



def get_args():
    parser = argparse.ArgumentParser("DF_CLIP", add_help=True)
    # path
    parser.add_argument("--dataset_root", type=str, default='./datasets', help="model configs")
    parser.add_argument("--scenario", type=str, default='cross_dataset', help="scenario")
    parser.add_argument("--training_sets", type=str, nargs="+", default=['FF++'], help="training datasets")
    parser.add_argument("--test_sets", type=str, nargs="+", default=['FFIW', 'Celeb-DF', 'DFDC-P', 'DFD'],
                        help="test datasets")
    parser.add_argument("--learning_rate", type=float, default=0.00004, help="learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="num_workers")
    parser.add_argument("--log_steps", type=int, default=20, help="logging step")

    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    parser.add_argument("--out_indices", type=int, nargs="+",
                        default=[24], help="features used in image and text encoder")
    parser.add_argument("--total_d_layer", type=int, default=11,
                        help="")
    parser.add_argument("--num_tokens", type=int, default=20,
                        help="the length of the deep prompt tuning")
    parser.add_argument("--use_global", default=True, action="store_false",
                        help="Whether to use global visual features in the CEP module")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="head number in the CEE module")
    parser.add_argument("--text_prompt_len", type=int, default=2,
                        help="head number in the CEE module")
    args = parser.parse_args()

    args = parser.parse_args()

    return args

class Trainer(object):

    def __init__(self, args, device, checkpoint):
        self.args = args
        self.device = device
        self.train_dataloader, self.test_dataloaders= data_loader(args)

        self.model = load_model(args, device)
        print('checkpoint', checkpoint)

        with open(checkpoint, 'rb') as opened_file:
            state_dict = torch.load(opened_file, map_location=device)

        u, w = self.model.load_state_dict(state_dict, False)
        print(u, w, 'are misaligned params in the model')


    def test(self):
        self.model.eval()

        for name, test_data_loader in self.test_dataloaders.items():
            img_pred, img_label = [], []
            for img, label in test_data_loader:
                img = img.to(self.device)
                label = label.to(self.device)
                with torch.no_grad():
                  forgery_score = self.model(img)

                img_label.append(label)
                img_pred.append(forgery_score)

            img_pred = torch.cat(img_pred, dim=0)
            img_label = torch.cat(img_label, dim=0)
            acc, auc, eer, precision, recall, f1, ap = dec_metrics(img_pred, img_label)

            with open(f'./cep_clip/{name}.txt', 'a', encoding='utf-8') as f:
                log_str = f'Dataset {name}, acc: {acc}, auc: {auc}, EER: {eer}, Precision: {precision}, Recall: {recall}, f1: {f1}, ap: {ap} \n'
                print(log_str)
                f.write(log_str)


if __name__ == "__main__":
    device = 'cuda:0'
    args = get_args()
    checkpoints = './checkpoints/best.pt'

    trainer = Trainer(args=args, device=device, checkpoint=checkpoint)
    with torch.no_grad():
        trainer.test()

