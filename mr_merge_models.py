import argparse
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
from transformers import ViTImageProcessor, ViTForImageClassification
from peft import LoraConfig, get_peft_model

from dataset import load_multidomain_dataset, ImageCollator

def _use_subsets(ds, size):
    print('############################################')
    print('YOU ARE GOING TO USE A SUBSET OF THE DATASET')
    print('########## ONLY FOR DEBUG PURPOSES #########')
    print('############### ARE YOU SURE? ##############')
    breakpoint()

    return Subset(ds, list(range(size)))

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

def training_loop(model, train_loader, opt, loss_fn, epochs, device):
    for e in range(epochs):
        correct_preds = 0
        for X, y in tqdm(train_loader):
            X = X.to(device)
            y = y.to(device)

            opt.zero_grad()
            y_pred = model(X).logits
            loss = loss_fn(y_pred, y)
            loss.backward()
            opt.step()

            # update to compute the accuracy
            correct_preds += torch.sum(torch.argmax(y_pred, dim=1).cpu() == y.cpu()).item()

        acc = correct_preds/len(train_loader.dataset)
        print(f'Accuracy at epoch {e}: {acc}')


def run(args):
    # load all dataset
    # load all the different domains
    # domain names are fixed, those are the currently supported datasets
    datasets = []
    for ds_name in ['domain_net', 'PACS', 'VLCS', 'office_home']:
        datasets.append(load_multidomain_dataset(args.ds_root, ds_name))
        print(f'{ds_name} dataset loaded!')

    # load the processor
    # it is the same for all models
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    collator = ImageCollator(processor)
    print('ViT Processor loaded!')

    # train an adapter for each auxiliary task
    # merge the adapter to the model -> get an updated model after each task
    # merge the models
    aux_models = {}
    for aux_ds in datasets:
        if aux_ds.name == args.target_ds: break # train only on auxiliary tasks, not on the target dataset

        print(f'Auxiliary pre-training on {aux_ds.name}')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=aux_ds.n_classes)

        # we want to use the entire dataset for training
        # hence, concat all the domains to create the train set
        train_domains = ConcatDataset([d for d in aux_ds.domains])
        train_domains = _use_subsets(train_domains, 8)
        train_loader = DataLoader(train_domains, collate_fn=collator, batch_size=args.batch_size)

        # define LoRA model
        # ??? Should I train the classifier too for auxiliary tasks ???
        config = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.1, target_modules=['query', 'value']) #, modules_to_save=['classifier'])
        lora_model = get_peft_model(model, config)
        lora_model.print_trainable_parameters()

        # finetune on train set
        opt = SGD(lora_model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()
        lora_model.to(args.device)
        lora_model.train()

        training_loop(lora_model, train_loader, opt, loss_fn, args.epochs, args.device)
        print(f'Finished the training on {aux_ds.name}')
        
        # merge the base model and the adapters
        aux_models[aux_ds.name] = lora_model.merge_and_unload()
        print('Merged model and adapter!')

    assert len(aux_models) == len(datasets) - 1, "There are less models than expected"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_root', type=str, default='/leonardo_scratch/fast/IscrC_FoundCL/projects/cl-collab/ModelRatatouille/lquarant/domainbed/data/')
    parser.add_argument('--target_ds', type=str, default='PACS')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    run(args)
