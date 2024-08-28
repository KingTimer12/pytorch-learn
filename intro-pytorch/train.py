import torch
import torch.nn as nn
import torch.optim as optim
from model import device

# Hyperparameters para treinar o modelo
"""
We define the following hyperparameters for training:
 - Number of Epochs - the number times the entire training dataset is passed through the network. 
 - Batch Size - the number of data samples seen by the model in each epoch. 
                Iterates over the number of batches needed to complete an epoch.
 - Learning Rate - the size of steps that the model matches as it searches 
                for the best weights that will produce a higher model accuracy. 
                Smaller values means the model will take a longer time to find the best weights. 
                Larger values may result in the model stepping over and missing the best weights, 
                which yields unpredictable behavior during training.
"""
epochs = 50
batch_size = 64
learning_rate = 0.01 # 1e-3

loss_fn = nn.CrossEntropyLoss()

def train(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            
def test(dataloader, model):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def run(model, train_dataloader, test_dataloader):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for i in range(epochs):
        print(f"Epoch {i+1}\n-------------------------------")
        train(train_dataloader, model, optimizer)
        test(test_dataloader, model)
    torch.save(model.state_dict(), "data/model.pth")
    print("Done!")