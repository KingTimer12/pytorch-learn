import torch.nn as nn
from torch.nn.functional import log_softmax, relu


class ConvModel(nn.Module):
    def __init__(self) -> None:
        super(ConvModel, self).__init__()
        """
        Convolutional layers are defined using `nn.Conv2d` construction. We need to specify the following:
        * `in_channels` - number of input channels. 
        In our case we are dealing with a grayscale image, thus number of input channels is 1. 
        Color image has 3 channels (RGB).
        * `out_channels` - number of filters to use. 
        We will use 9 different filters, which will give the network plenty of 
        opportunities to explore which filters work best for our scenario.
        * `kernel_size` is the size of the sliding window. 
        Usually 3x3 or 5x5 filters are used. 
        The choice of filter size is usually chosen by experiment, 
        that is by trying out different filter sizes and comparing resulting accuracy.
        """
        # O tamanho da entrada é 28x28, mas
        # o tamanho espacial da saída após a convolução é 24x24.
        # Isso ocorre porque a convolução com um kernel de tamanho 5x5
        # resulta em uma redução de 2 pixels em cada dimensão.
        # Portanto, a saída da convolução é |out_channels|x24x24. (9x24x24)
        self.conv = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=5)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(9 * 24 * 24, 10)
    
    def forward(self, x):
        x = relu(self.conv(x))
        x = self.flatten(x)
        x = self.linear(x)
        return log_softmax(x, dim=1)
        
class MultiLayerCNN(nn.Module):
    def __init__(self) -> None:
        super(MultiLayerCNN, self).__init__()
        """
        Note a few things about the definition:
        * Instead of using `Flatten` layer, 
        we are flattening the tensor inside `forward` function using `view` function, 
        which is similar to `reshape` function in numpy. 
        Since flattening layer does not have trainable weights, 
        it is not required that we create a separate layer instance within our class - 
        we can just use a function from `torch.nn.functional` namespace.
        * We use just one instance of pooling layer in our model, 
        also because it does not contain any trainable parameters, 
        and thus one instance can be effectively reused.
        * The number of trainable parameters (~8.5K) is dramatically smaller than in previous cases 
        (80K in Perceptron, 50K in one-layer CNN). 
        This happens because convolutional layers in general have few parameters, 
        independent of the input image size. 
        Also, due to pooling, dimensionality of the image is significantly reduced before applying final dense layer. 
        Small number of parameters have positive impact on our models, 
        because it helps to prevent overfitting even on smaller dataset sizes.
        """
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.linear = nn.Linear(320, 10)
    
    def forward(self, x):
        x = self.pool(relu(self.conv1(x)))
        x = self.pool(relu(self.conv2(x)))
        x = x.view(-1, 320)
        return log_softmax(self.linear(x), dim=1)

class LeNet(nn.Module):
    def __init__(self) -> None:
        super(LeNet, self).__init__()
        """
        The LeNet architecture consists of two sets of convolutional and average pooling layers, 
        followed by a flattening convolutional layer, 
        two fully connected layers, and finally a softmax classifier.
        """
        # In é 3 pq a entrada é uma imagem colorida (RBG)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)
        self.flat = nn.Flatten()
    
    def forward(self, x):
        x = self.pool(relu(self.conv1(x)))
        x = self.pool(relu(self.conv2(x)))
        x = relu(self.conv3(x))
        x = self.flat(x)
        x = relu(self.linear1(x))
        return self.linear2(x)