import os
import torch
import torchaudio

import matplotlib.pyplot as plt

from train import run, test_dataloader, device
from model import CNNet
from torchinfo import summary

if not os.path.exists('model.pth'):
    run()

model = CNNet().to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()
summary(model, input_size=(15, 3, 201, 81))

test_loss, correct = 0, 0
class_map = ['no', 'yes']

with torch.no_grad():
    for batch, (X, Y) in enumerate(test_dataloader):
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        print("Predicted:\nvalue={}, class_name= {}\n".format(pred[0].argmax(0),class_map[pred[0].argmax(0)]))
        print("Actual:\nvalue={}, class_name= {}\n".format(Y[0],class_map[Y[0]]))
        break