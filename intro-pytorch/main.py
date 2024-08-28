import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from model import NeuralNetwork, device
from train import run

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
classes = [item for item in labels_map.values()]
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = int(torch.randint(len(training_data), size=(1,)).item())
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display image and label.
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# label_name = list(labels_map.values())[label]
# print(f"Label: {label_name}")
# plt.imshow(img, cmap="gray")
# plt.title(label_name)
# plt.show()

model = NeuralNetwork().to(device)
# print(model)
# print(f"First Linear weights:\n {model.linear_relu_stack[0].weight}")
# print(f"First Linear biases:\n {model.linear_relu_stack[0].bias}")

if not os.path.exists('data/model.pth'):
    run(model, train_dataloader, test_dataloader) # Accuracy: 79.2%, Avg loss: 0.008756

model.load_state_dict(torch.load('data/model.pth', weights_only=True))
model.eval()

def pred(i):
    x, y = test_data[i][0], test_data[i][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}", Correct: {predicted == actual}')
        return predicted == actual

test_size = 50
correct, wrong = 0, 0
for i in range(test_size):
    if pred(i):
        correct += 1
    else:
        wrong += 1

correct_percent = (correct / test_size) * 100
print(f'Correct: {correct}, Wrong: {wrong} \nAccuracy: {(correct_percent):>0.1f}%')