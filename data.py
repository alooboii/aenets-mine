import os
import torch
import numpy as np
import tarfile
import requests
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageDraw


# Imagenette download logic (without fastai)
def download_imagenette(root="./data", dest_tgz="imagenette2-160.tgz", extract_dir="imagenette"):
    os.makedirs(root, exist_ok=True)
    tgz_path = os.path.join(root, dest_tgz)
    extract_path = os.path.join(root, extract_dir)
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    if not os.path.exists(tgz_path):
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))
        with open(tgz_path, 'wb') as file, tqdm(
            desc=tgz_path,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    if not os.path.exists(extract_path):
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(extract_path)
    return (
        os.path.join(extract_path, "imagenette2-160", "train"),
        os.path.join(extract_path, "imagenette2-160", "val"),
    )

# Transform classes
class JitterTransform:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)
    def __call__(self, img):
        return self.jitter(img)

class EdgeTransform:
    def __call__(self, img):
        return img.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")

class AddNoiseTransform:
    def __call__(self, img):
        np_img = np.array(img) / 255.0
        noise = np.random.normal(0, 0.1, np_img.shape)
        np_img = np.clip(np_img + noise, 0, 1)
        return Image.fromarray((np_img * 255).astype(np.uint8))

# Spurious overlay wrapper for training set
def add_label_card(img: Image.Image, label: int):
    card = Image.new('RGB', (16, 16), 'white')
    draw = ImageDraw.Draw(card)
    draw.text((1, 1), str(label), fill='black')
    w, h = img.size
    img.paste(card, (0, h - 16))
    return img

class SpuriousDataset(Dataset):
    def __init__(self, base_dataset: Dataset, transform=None):
        self.base = base_dataset
        self.transform = transform
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, label = self.base[idx]
        img = add_label_card(img, label)
        if self.transform:
            img = self.transform(img)
        return img, label

# Compose standard transforms
def get_transforms(edge=False, jitter=False, noise=False, model_family="cnn"):
    use_imagenet_stats = model_family == "vit"
    if use_imagenet_stats:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        resize_ops = [transforms.Resize(256), transforms.CenterCrop(224)]
    else:
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        resize_ops = [transforms.Resize((224, 224))]

    if edge:
        pipeline = [
            *resize_ops,
            EdgeTransform(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    elif jitter:
        pipeline = [
            *resize_ops,
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    elif noise:
        pipeline = [
            *resize_ops,
            AddNoiseTransform(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    else:
        pipeline = [
            *resize_ops,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    return transforms.Compose(pipeline)

# Load dataset with optional spurious label cards
def get_dataset(name="CIFAR100", root="./data", download=True,
                jitter=False, edge=False, noise=False, set_spurious=False, model_family="cnn"):
    name = name.upper()
    transform = get_transforms(edge, jitter, noise, model_family=model_family)

    if name in ["CIFAR10", "CIFAR100"]:
        cls = torchvision.datasets.CIFAR10 if name == "CIFAR10" else torchvision.datasets.CIFAR100
        train_raw = cls(root, train=True, download=download, transform=None)
        test_raw  = cls(root, train=False, download=download, transform=None)
        if set_spurious:
            train = SpuriousDataset(train_raw, transform)
        else:
            train = cls(root, train=True, download=download, transform=transform)
        test  = cls(root, train=False, download=download, transform=transform)
        num_classes = len(train_raw.classes)

    elif name == "IMAGENETTE":
        train_dir, val_dir = download_imagenette(root=root)
        train_raw = torchvision.datasets.ImageFolder(train_dir, transform=None)
        test_raw  = torchvision.datasets.ImageFolder(val_dir,   transform=None)
        if set_spurious:
            train = SpuriousDataset(train_raw, transform)
        else:
            train = torchvision.datasets.ImageFolder(train_dir, transform=transform)
        test = torchvision.datasets.ImageFolder(val_dir, transform=transform)
        num_classes = len(train_raw.classes)

    elif name == "FOOD101":
        train_raw = torchvision.datasets.Food101(root=root, split="train", download=download, transform=None)
        test_raw  = torchvision.datasets.Food101(root=root, split="test",  download=download, transform=None)
        if set_spurious:
            train = SpuriousDataset(train_raw, transform)
        else:
            train = torchvision.datasets.Food101(root=root, split="train", download=download, transform=transform)
        test  = torchvision.datasets.Food101(root=root, split="test",  download=download, transform=transform)
        num_classes = 101

    elif name == "CUSTOM":
        train_dir = os.path.join(root, 'train')
        val_dir   = os.path.join(root, 'valid')
        train_raw = torchvision.datasets.ImageFolder(train_dir, transform=None)
        test_raw  = torchvision.datasets.ImageFolder(val_dir,   transform=None)
        if set_spurious:
            train = SpuriousDataset(train_raw, transform)
        else:
            train = torchvision.datasets.ImageFolder(train_dir, transform=transform)
        test = torchvision.datasets.ImageFolder(val_dir, transform=transform)
        num_classes = len(train_raw.classes)

    else:
        raise ValueError(f"Unknown dataset: {name}. Choose CIFAR10, CIFAR100, IMAGENETTE, FOOD101, or CUSTOM.")

    return train, test, num_classes

# DataLoader function
def get_dataloaders(dataset="CIFAR100", batch_size=128, root="./data",
                    jitter=False, edge=False, noise=False, set_spurious=False, model_family="cnn"):
    train_set, test_set, num_classes = get_dataset(dataset, root, True,
                                                   jitter, edge, noise, set_spurious, model_family)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader, num_classes

# HANDLING THE PACS DATASET
# def get_pacs_dataloaders(batch_size=128, test_domain=None, 
#                         jitter=False, edge=False, noise=False, set_spurious=False):
#     """
#     Load PACS dataset using Deep Lake and return PyTorch DataLoaders.
    
#     Args:
#         batch_size (int): Batch size for DataLoaders
#         test_domain (str, optional): Domain to use as test set. Options: 'art_painting', 'cartoon', 'photo', 'sketch'
#                                    If None, all domains are included in both train and test splits
#         jitter (bool): Apply color jitter augmentation
#         edge (bool): Apply edge detection transform
#         noise (bool): Add noise to images
#         set_spurious (bool): Add spurious label cards to training images
    
#     Returns:
#         tuple: (train_loader, test_loader, num_classes)
#     """
    
#     # Define transforms
#     transform = get_transforms(edge, jitter, noise)
    
#     train_ds = deeplake.load("hub://activeloop/pacs-train")
#     test_ds = deeplake.load("hub://activeloop/pacs-val")
    
#     # Print dataset info for debugging
#     print(f"Dataset length: {len(train_ds) + len(test_ds)}")
#     print(f"Dataset tensors: {list(train_ds.tensors.keys())}")
    
#     # Check the first sample to understand structure
#     sample = test_ds[0]
    
#     # PACS has 4 domains and 7 classes
#     domains = ['art_painting', 'cartoon', 'photo', 'sketch']
#     num_classes = 7
    
#     # If test domain is specified, then we filter the train and test splits for the domains
#     if test_domain is not None:
#         if test_domain not in domains:
#             raise ValueError(f"test_domain must be one of {domains}")
        
#         # Filter the train set
#         train_doms = list(np.array(train_ds.domains.data()['text']).squeeze())
#         test_doms = list(np.array(test_ds.domains.data()['text']).squeeze())
        
#         train_indices = [i for i, domain in enumerate(train_doms) if domain != test_domain]
#         test_indices  = [i for i, domain in enumerate(test_doms)  if domain == test_domain]

#         train_ds = train_ds[train_indices]
#         test_ds = test_ds[test_indices]
    
#     # Convert to PyTorch datasets with transforms
#     class PACSDataset:
#         def __init__(self, deeplake_ds, transform=None, spurious=False):
#             self.ds = deeplake_ds
#             self.transform = transform
#             self.spurious = spurious
            
#         def __len__(self):
#             return len(self.ds)
            
#         def __getitem__(self, idx):
#             sample = self.ds[idx]
            
#             # Handle different ways the image might be stored
#             if hasattr(sample, 'images'):
#                 if hasattr(sample.images, 'pil'):
#                     image = sample.images.pil()
#                 elif hasattr(sample.images, 'numpy'):
#                     # Convert numpy array to PIL
#                     img_array = sample.images.numpy()
#                     if img_array.max() <= 1.0:
#                         img_array = (img_array * 255).astype(np.uint8)
#                     image = Image.fromarray(img_array)
#                 else:
#                     # Direct tensor access
#                     img_tensor = sample.images
#                     if isinstance(img_tensor, torch.Tensor):
#                         # Convert tensor to PIL
#                         if img_tensor.dim() == 3 and img_tensor.shape[0] in [1, 3]:
#                             # CHW format
#                             img_array = img_tensor.permute(1, 2, 0).numpy()
#                         else:
#                             img_array = img_tensor.numpy()
                        
#                         if img_array.max() <= 1.0:
#                             img_array = (img_array * 255).astype(np.uint8)
#                         image = Image.fromarray(img_array)
#                     else:
#                         # Numpy array
#                         img_array = np.array(img_tensor)
#                         if img_array.max() <= 1.0:
#                             img_array = (img_array * 255).astype(np.uint8)
#                         image = Image.fromarray(img_array)
#             else:
#                 # Try other common field names
#                 for field_name in ['image', 'img', 'data']:
#                     if hasattr(sample, field_name):
#                         image_data = getattr(sample, field_name)
#                         if hasattr(image_data, 'pil'):
#                             image = image_data.pil()
#                         else:
#                             # Convert to PIL as above
#                             img_array = np.array(image_data)
#                             if img_array.max() <= 1.0:
#                                 img_array = (img_array * 255).astype(np.uint8)
#                             image = Image.fromarray(img_array)
#                         break
#                 else:
#                     raise ValueError(f"Could not find image data in sample. Available keys: {list(sample.keys())}")
            
#             # Handle labels
#             if hasattr(sample, 'labels'):
#                 if hasattr(sample.labels, 'numpy'):
#                     label = sample.labels.numpy().item()
#                 else:
#                     label = int(sample.labels)
#             elif hasattr(sample, 'label'):
#                 if hasattr(sample.label, 'numpy'):
#                     label = sample.label.numpy().item()
#                 else:
#                     label = int(sample.label)
#             else:
#                 # Try to find label field
#                 for field_name in ['target', 'class', 'y']:
#                     if hasattr(sample, field_name):
#                         label_data = getattr(sample, field_name)
#                         if hasattr(label_data, 'numpy'):
#                             label = label_data.numpy().item()
#                         else:
#                             label = int(label_data)
#                         break
#                 else:
#                     raise ValueError(f"Could not find label data in sample. Available keys: {list(sample.keys())}")
            
#             # Add spurious correlation if requested
#             if self.spurious:
#                 image = add_label_card(image, label)
            
#             # Apply transforms
#             if self.transform:
#                 image = self.transform(image)
                
#             return image, label
    
#     # Create PyTorch datasets
#     if set_spurious:
#         train_dataset = PACSDataset(train_ds, transform, spurious=True)
#     else:
#         train_dataset = PACSDataset(train_ds, transform, spurious=False)
    
#     test_dataset = PACSDataset(test_ds, transform, spurious=False)
    
#     # Create DataLoaders
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=batch_size, 
#         shuffle=True, 
#         num_workers=4, 
#         pin_memory=True
#     )
    
#     test_loader = DataLoader(
#         test_dataset, 
#         batch_size=batch_size, 
#         shuffle=False, 
#         num_workers=4, 
#         pin_memory=True
#     )
    
    # return train_loader, test_loader, num_classes
