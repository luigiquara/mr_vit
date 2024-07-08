import argparse
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, PeftModel

from dataset import load_multidomain_dataset, ImageCollator

class ViT_and_Classifier(nn.Module):
    def __init__(self, vit, n_classes):
        super(ViT_and_Classifier, self).__init__()

        self.vit = vit
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(vit.config.hidden_size, n_classes)

    def forward(self, pixel_values):
        output = self.vit(pixel_values = pixel_values)
        logits = self.classifier(self.dropout(output.last_hidden_state[:,0]))
        return logits

def _use_subsets(ds, size):
    print('############################################')
    print('YOU ARE GOING TO USE A SUBSET OF THE DATASET')
    print('######## ONLY FOR DEBUG PURPOSES ###########')
    print('############## ARE YOU SURE? ###############')
    print('############################################')

    breakpoint()

    return Subset(ds, list(range(size)))

def training_loop(model, train_loader, opt, loss_fn, epochs, device):
    for e in range(epochs):
        correct_preds = 0
        for X, y in tqdm(train_loader):
            X = X.to(device)
            y = y.to(device)

            opt.zero_grad()
            #y_pred = model(X).logits
            y_pred = model(X)
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

    # load the model and the processor
    # it is the same for all models
    vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    collator = ImageCollator(processor)
    print('ViT Processor loaded!')

    # train an adapter for each auxiliary task
    # merge the adapter to the model -> get an updated model after each task
    # merge the models
    loras = {}
    for aux_ds in datasets:
        if aux_ds.name == args.target_ds: continue # train only on auxiliary tasks, not on the target dataset

        print(f'Auxiliary pre-training on {aux_ds.name}')

        # we want to use the entire dataset for training
        # hence, concat all the domains to create the train set
        train_domains = ConcatDataset([d for d in aux_ds.domains.values()])
        #train_domains = _use_subsets(train_domains, 8)
        train_loader = DataLoader(train_domains, collate_fn=collator, batch_size=args.batch_size)

        # define LoRA model
        # ??? Should I train the classifier too for auxiliary tasks ???
        config = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.1, target_modules=['query', 'value']) #, modules_to_save=['classifier'])

        # the model has already some adapters
        # simply add new adapter and set it as the active one
        if isinstance(vit, PeftModel):
            print('Adding new adapter')
            vit.add_adapter(f'{aux_ds.name}_lora', config)
            vit.set_adapter(f'{aux_ds.name}_lora')

        # it's the first time the model gets an adapter
        # so you have to create the PeftModel
        else:
            print('First time with an adapter')
            vit = get_peft_model(vit, config, adapter_name=f'{aux_ds.name}_lora')
        vit.print_trainable_parameters()

        # define the model with classifier
        model = ViT_and_Classifier(vit, aux_ds.n_classes)

        # finetune on train set
        opt = SGD(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()
        model.to(args.device)
        model.train()

        training_loop(model, train_loader, opt, loss_fn, args.epochs, args.device)
        print(f'Finished the training on {aux_ds.name}')
        
        # merge the base model and the adapters
        loras[aux_ds.name] = model
        print('Adapter saved!')

    assert len(loras) == len(datasets) - 1, "There are less models than expected"

    # Merge LoRA adapters
    adapters = [f'{aux_ds.name}_lora' for aux_ds in datasets if aux_ds.name != args.target_ds]
    weights = [1.0 for _ in range(len(datasets))]
    adapter_name = 'merge'
    combination_type = 'dare_ties'
    density = 0.2 #from huggingface documentation

    vit.add_weighted_adapter(adapters, weights, adapter_name, combination_type, density=density)
    merged_vit = vit.merge_and_unload(progressbar=True, adapter_names=[adapter_name]) # new adapter merged with the base model

    # set new classifier for the downstream dataset
    for d in datasets:
        if d.name == args.target_ds:
            target_ds = d
            break
    model = ViT_and_Classifier(merged_vit, target_ds.n_classes)

    # at each time, set one domain as the test domain
    # then average the accuracies
    total_acc = 0
    for test_name, test_domain in target_ds.domains.items():
        print(f'Testing on {test_name}')
        train_domains = ConcatDataset([d for d in target_ds.domains.values() if d != test_domain])
        #train_domains = _use_subsets(train_domains, 8)
        #test_domain = _use_subsets(test_domain, 8)
        train_loader = DataLoader(train_domains, collate_fn=collator, batch_size=args.batch_size)
        test_loader = DataLoader(test_domain, collate_fn=collator, batch_size=args.batch_size)

        # new lora to finetune on training domains
        config = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.1, target_modules=['query', 'value']) #, modules_to_save=['classifier'])
        model.vit = get_peft_model(model.vit, config)
        #model.vit.add_adapter('train_domains', config)
        #model.set_adapter('train_domains')

        # finetuning on test dataset, training domains
        opt = SGD(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()
        model.to(args.device)
        model.train()

        training_loop(model, train_loader, opt, loss_fn, args.epochs, args.device)

        # test on test dataset, test domain
        model.eval()
        correct_preds = 0

        for X, y in test_loader:
            X = X.to(args.device)

            with torch.no_grad():
                y_pred = model(X)

            correct_preds += torch.sum(torch.argmax(y_pred, dim=1).cpu() == y.cpu()).item()

        acc = correct_preds / len(test_loader.dataset)
        total_acc += acc
        print(f'Accuracy when testing on {test_name}: {acc}')

    avg_acc = total_acc / len(target_ds.domains)
    print(f'Average test accuracy: {avg_acc}')

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
