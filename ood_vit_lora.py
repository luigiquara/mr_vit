import argparse
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
from transformers import ViTImageProcessor, ViTForImageClassification
from peft import LoraConfig, get_peft_model

from dataset import load_multidomain_dataset

class ImageCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        inputs = self.processor([img for img, label in batch], return_tensors='pt')
        inputs['label'] = [label for _, label in batch]
        return inputs['pixel_values'], torch.LongTensor(inputs['label'])

def forward_pass(model, loader, device):
    '''Perform a forward pass on the model. Return the output features

    Parameters
    ----------
    model:
        A ML model
    loader: torch.utils.data.DataLoader
        The DataLoader to iterate over a given dataset
    device: str
        The PyTorch device to use

    Returns
    -------
    features: torch.Tensor
        The output features from the model. Shape (batch_size, output_size)
    target: torch.Tensor
        Tensor with the labels. Shape (batch_size, 1)
    '''

    features = []
    targets = []
    with torch.no_grad():
        for X, y in tqdm(loader):
            X = X.to(device)
            model.to(device)

            output = model(X)
            targets.append(y)
    
    return torch.cat(features), torch.cat(targets)

def run(args):
    # load dataset
    # load all the different domains
    domains, n_classes = load_multidomain_dataset(args.ds_root, args.ds_name)
    print(f'{args.ds_name} dataset loaded!')

    # load the model and processor
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=n_classes)
    print('ViT loaded!')

    collator = ImageCollator(processor) # apply preprocessing when collating batches

    # at each time, select a domain as test_domain
    # train on all the other ones, evaluate on test_domain
    total_acc = 0
    for test_name, test_domain in domains.items():
        print(f'Testing on {test_name}')
        # concat all the other domains to create the train set
        train_domains = ConcatDataset([d for d in domains.values() if d != test_domain])

        #train_domains = Subset(train_domains, list(range(8)))
        #test_domain = Subset(test_domain, list(range(4)))
        
        train_loader = DataLoader(train_domains, collate_fn=collator, batch_size=args.batch_size)
        test_loader = DataLoader(test_domain, collate_fn=collator, batch_size=args.batch_size)

        # define LoRA model
        config = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.1, target_modules=['query', 'value'], modules_to_save=['classifier'])
        lora_model = get_peft_model(model, config)
        lora_model.print_trainable_parameters()

        # finetune on train set
        opt = SGD(lora_model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()
        lora_model.to(args.device)
        lora_model.train()
        
        for e in range(args.epochs):
            correct_preds = 0
            for X, y in tqdm(train_loader):
                X = X.to(args.device)
                y = y.to(args.device)

                opt.zero_grad()
                y_pred = lora_model(X).logits
                loss = loss_fn(y_pred, y)
                loss.backward()
                opt.step()

                correct_preds += torch.sum(torch.argmax(y_pred, dim=1).cpu() == y.cpu()).item()

            acc = correct_preds/len(train_loader.dataset)
            print(f'Accuracy at epoch {e}: {acc}')

        
        # evaluation on test domain
        lora_model.to(args.device)
        lora_model.eval()
        correct_preds = 0

        for X, y in tqdm(test_loader):
            X = X.to(args.device)

            with torch.no_grad():
                y_pred = lora_model(X).logits

            correct_preds += torch.sum(torch.argmax(y_pred, dim=1).cpu() == y.cpu()).item()

        acc = correct_preds / len(test_loader.dataset)
        total_acc += acc
        print(f'Accuracy on test domain {test_name}: {acc}')

    avg_acc = total_acc / len(domains)
    print(f'Avg accuracy on all test domains: {avg_acc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', type=str)
    parser.add_argument('--ds_root', type=str, default='/leonardo_scratch/fast/IscrC_FoundCL/projects/cl-collab/ModelRatatouille/domainbed/data/')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    run(args)
