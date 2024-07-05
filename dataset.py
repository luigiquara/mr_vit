from dataclasses import dataclass

from PIL import ImageFile
from torchvision.datasets import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True

@dataclass
class MD_Dataset:
    n_classes: int
    name: str
    domains: dict

class ImageCollator:
    '''Prepare the batch by pre-processing the images and selecting the right stuff'''

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        inputs = self.processor([img for img, label in batch], return_tensors='pt')
        inputs['label'] = [label for _, label in batch]
        return inputs['pixel_values'], torch.LongTensor(inputs['label'])

def load_multidomain_dataset(ds_root, ds_name):
    '''Load datasets with multiple domains, for OOD generalization.
    The supported datasets are the ones from ModelRatatouille: PACS, VLCS, OfficeHome, TerraInc, DomainNet
    
    Parameters
    ---------
    ds_root: str
        The path to the directory containing all the datasets
    ds_name: str
        The name of the desired dataset. Must be in [PACS, VLCS, OfficeHome, TerraInc, DomainNet]

    Return
    ------
    n_classes: int
        Number of classes of the specified dataset
    domains: {ds_name: str, ds: ImageFolder}
        Dictionary containing all the domains of the specified dataset
    '''

    ds_path = ds_root + ds_name

    if ds_name == 'PACS':
        domain_names = ['photo', 'art_painting', 'cartoon', 'sketch']
        n_classes = 7
    elif ds_name == 'VLCS':
        domain_names = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
        n_classes = 5
    elif ds_name == 'domain_net':
        domain_names = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        n_classes = 345
    elif ds_name == 'office_home':
        domain_names = ['Art', 'Clipart', 'Product', 'Real World']
        n_classes = 65

    elif ds_name == 'terra_incognita': raise NotImplementedError
    else: raise Exception('dataset not found')

    domains = {}
    # load all the domains
    for d in domain_names: domains[d] = ImageFolder(ds_path + f'/{d}')

    dataset = MD_Dataset(n_classes, ds_name, domains)
    return dataset
