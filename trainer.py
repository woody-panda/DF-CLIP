from data_utils import data_loader
from model import *
import torch
from metrics import dec_metrics


def freeze_stages(model, exclude_keys):

    parameter_prompt_dict = {}
    for n, m in model.named_parameters():
        if any(key in n for key in exclude_keys):
            print(n,m.size())
            parameter_prompt_dict[n] = m
        else:
            m.requires_grad = False
    return parameter_prompt_dict


class Trainer(object):
    def __init__(self, args, device):
        self.args = args
        self.epochs = args.epochs
        self.log_steps = args.log_steps
        self.device = device
        self.train_dataloader, self.test_dataloaders= data_loader(args)

        self.model = load_model(args, device)

        exclude_keys = ['prompt', 'g_project', 'context']
        parameters_optim = freeze_stages(self.model, exclude_keys=exclude_keys)
        parameter_list = [value for key,value in parameters_optim.items()]
        self.optimizer = torch.optim.Adam(params=parameter_list, lr=args.learning_rate)
        self.dec_criterion = nn.CrossEntropyLoss()

    def train(self):
        for epoch in range(1, self.epochs):
            self.model.train()
            for step, (img, label) in enumerate(self.train_dataloader):
                img = img.to(self.device)
                label = label.to(self.device)
                
                forgery_score = self.model(img)

                overall_loss = self.dec_criterion(forgery_score, label)
                try:
                    acc, auc, eer, precision, recall, f1, ap = dec_metrics(forgery_score, label)
                    if step % self.log_steps == 0:
                        with open('./training.log', 'a', encoding='utf-8') as f:
                            log_str = f'Epoch: {epoch}, Step: {step + 1}, acc: {acc}, auc: {auc}, eer: {eer}, precision: {precision}, recall: {recall}, ap: {ap}, f1: {f1}, Loss: {overall_loss}  \n'

                            f.write(log_str)
                except:
                    pass
                self.optimizer.zero_grad()
                overall_loss.backward()
                self.optimizer.step()
                
            self.test()
            save_path = f'./checkpoints/{epoch}.pt'
            torch.save(self.model.state_dict(), save_path)

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

            with open('./test.log', 'a', encoding='utf-8') as f:
                log_str = f'Dataset {name}, acc: {acc}, auc: {auc}, eer: {eer}, precision: {precision}, recall: {recall}, ap: {ap}, f1: {f1} \n'

                f.write(log_str)

