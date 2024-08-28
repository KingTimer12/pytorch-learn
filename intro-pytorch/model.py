import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Our neural network is composed of the following:

- The input layer with 28x28 or 784 features/pixels.
- The first linear module takes the input 784 features and transforms it to a hidden layer with 512 features.
- The ReLU activation function will be applied in the transformation.
- The second linear module takes 512 features as input from the first hidden layer and transforms it to the next hidden layer with 512 features.
- The ReLU activation function will be applied in the transformation.
- The third linear module take 512 features as input from the second hidden layer and transforms those features to the output layer with 10, which is the number of classes.
- The ReLU activation function will be applied in the transformation.
"""
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        """
        We initialize the `nn.Flatten` layer to convert each 2D 28x28 image into a contiguous array of 784 pixel 
        values, that is, the minibatch dimension (at dim=0) is maintained. 
        Each of the pixels are passed to the input layer of the neural network. 
        """
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # nn.Linear(784, 512) | 784 input features, 512 output features
            nn.ReLU(),
            nn.Linear(512, 512), # 512 input features, 512 output features
            nn.ReLU(),
            nn.Linear(512, 10), # 512 input features, 10 output features
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Exemplo de entedimento
# - Entra uma imagem de 28x28 pixels -
input_image = torch.rand(3,28,28)
# print(input_image.size())
# - A imagem é convertida num array de 784 pixels -
# - graças ao nn.Flatten                          -
flatten = nn.Flatten()
# flat_image = flatten(input_image)
# print(flat_image.size())
# - A imagem é passada para camada linear      -
# - Lá, a imagem é transformada em 20 features -
# - Onde seus valores são calculados           -
# - x = weight * input + bias                  -
layer1 = nn.Linear(in_features=28*28, out_features=20)
# hidden1 = layer1(flat_image)
# print(hidden1.size())
# Por fim, o resultado é passado para um função de ativação -
# No caso, a ReLU                                           -
# print(f"Before ReLU: {hidden1}\n\n")
# hidden1 = nn.ReLU()(hidden1)
# print(f"After ReLU: {hidden1}")

# O nn.Sequential é um container que agrupa módulos em ordem
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
logits = seq_modules(input_image)
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
# print(pred_probab)