import glob
import io
import os
import zipfile

import requests
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, ImageFolder
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)


def download_data(root: str = 'data'):
    data_train = MNIST(root='data', train=True, transform=ToTensor(), download=True)
    data_test = MNIST(root='data', train=False, transform=ToTensor(), download=True)
    
    return data_train, data_test


def load_data(data_train, data_test, batch_size: int = 64):
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def download_cifar10(root: str = 'data'):
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_train = CIFAR10(root='data', train=True, transform=transform, download=True)
    data_test = CIFAR10(root='data', train=False, transform=transform, download=True)
    return data_train, data_test
  
def load_cifar10(data_train, data_test, batch_size: int = 64):
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def download_file(url, path):
    path = os.path.join(path, 'PetImages')
    if not os.path.exists(path):
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path)
        check_image_dir(os.path.join(path, 'PetImages', 'Cat', '*.jpg'))
        check_image_dir(os.path.join(path, 'PetImages', 'Dog', '*.jpg'))
    std_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(), 
            std_normalize])
    path = os.path.join(path, 'PetImages')
    dataset = ImageFolder(path, transform=trans)
    data_train, data_test = data.random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])
    return data_train, data_test

def check_image(fn):
    try:
        im = Image.open(fn)
        im.verify()
        return True
    except:
        return False
    
def check_image_dir(path):
    for fn in glob.glob(path):
        if not check_image(fn):
            print("Corrupt image: {}".format(fn))
            os.remove(fn)