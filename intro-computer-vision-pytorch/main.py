import os
import requests
import json

import torch.nn as nn
from torch.optim.sgd import SGD
from torchvision.models import vgg16, VGG16_Weights

from data import (
    download_cifar10,
    download_data,
    download_file,
    load_cifar10,
    load_data,
)
from models import ConvModel, LeNet, MultiLayerCNN
from train import SimpleTrain

from utils import classifier_images

# Carregar MNIST
train_data, test_data = download_data()
train_loader, test_loader = load_data(train_data, test_data, batch_size=128)

cnn_01 = "data/conv_model01.pth"

if not os.path.exists(cnn_01):
    model = ConvModel()
    trainer = SimpleTrain(model, train_loader, test_loader)
    trainer.run(epoch=5, model_path=cnn_01)
    
cnn_01 = "data/conv_model02.pth"

if not os.path.exists(cnn_01):
    model = MultiLayerCNN()
    trainer = SimpleTrain(model, train_loader, test_loader)
    trainer.run(epoch=5, model_path=cnn_01)

# Carregar CIFAR10
train_data, test_data = download_cifar10()
train_loader, test_loader = load_cifar10(train_data, test_data, batch_size=128)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

cnn_01 = "data/lenet.pth"

if not os.path.exists(cnn_01):
    model = LeNet()
    opt = SGD(model.parameters(), lr=0.001, momentum=0.9)
    trainer = SimpleTrain(model, train_loader, test_loader)
    trainer.run(epoch=15, model_path=cnn_01, optimizer=opt, loss_fn=nn.CrossEntropyLoss())

# Carregar Cat & Dogs
train_data, test_data = download_file(
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip", 
    "data")

vgg = vgg16(weights=VGG16_Weights.DEFAULT)
vgg.eval()

# sample_image = train_data[0][0].unsqueeze(0)
# print(sample_image)
class_map = json.loads(requests.get("https://raw.githubusercontent.com/MicrosoftDocs/pytorchfundamentals/main/computer-vision-pytorch/imagenet_class_index.json").text)
class_map = { int(k) : v for k,v in class_map.items() }

classifier_images(train_data, vgg, classes=class_map)