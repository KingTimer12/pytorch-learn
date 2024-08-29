import torch
import torch.nn as nn
from torch.optim.adam import Adam

from utils import device


class SimpleTrain:
  def __init__(self, model, train_loader, test_loader):
    self.model = model
    self.train_loader = train_loader
    self.test_loader = test_loader

  def run(
    self, epoch: int = 5, optimizer=None, learning_rate: float = 0.01, loss_fn=None, model_path: str = 'data/model.pth'
  ):
    for t in range(epoch):
      print(f'Epoch {t+1}\n-------------------------------')
      self.__train(optimizer, learning_rate, loss_fn)
      self.__val(loss_fn)
    torch.save(self.model.state_dict(), model_path)

  def __train(self, optimizer=None, learning_rate: float = 0.01, loss_fn=None):
    optimizer = optimizer or Adam(self.model.parameters(), lr=learning_rate)
    loss_fn = loss_fn or nn.NLLLoss()
    self.model.train()  # Muda para o modo de treino
    size = len(self.train_loader.dataset)
    for batch, (features, labels) in enumerate(self.train_loader):
      features, labels = features.to(device), labels.to(device)

      output = self.model(features)
      loss = loss_fn(output, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if batch % 100 == 0:
        loss, current = loss.item(), batch * len(features)
        print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

  # A.K.A. __test
  def __val(self, loss_fn=None):
    loss_fn = loss_fn or nn.NLLLoss()
    self.model.eval()  # Muda para o modo de avaliação
    size = len(self.test_loader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
      for features, labels in self.test_loader:
        features, labels = features.to(device), labels.to(device)
        output = self.model(features)
        test_loss += loss_fn(output, labels).item()
        correct += (output.argmax(1) == labels).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(
      f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n'
    )
