import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


def show_images(data, end: int = 7, classes=None):
    fig,ax = plt.subplots(1, end, figsize=(15,3))
    mn = min([data[i][0].min() for i in range(end)])
    mx = max([data[i][0].max() for i in range(end)])
    for i in range(end):
        ax[i].imshow(np.transpose((data[i][0]-mn)/(mx-mn),(1,2,0)))
        ax[i].axis('off')
        if classes:
            ax[i].set_title(classes[data[i][1]])
    plt.show()
    
def classifier_images(data, model, end: int = 7, classes=None):
    fig,ax = plt.subplots(1, end, figsize=(15,3))
    mn = min([data[i][0].min() for i in range(end)])
    mx = max([data[i][0].max() for i in range(end)])
    for i in range(end):
        im = data[i][0]
        im_transpose = np.transpose((im-mn)/(mx-mn),(1,2,0))
        res = model(im.unsqueeze(0).to(device)).to(device)
        ax[i].imshow(im_transpose)
        ax[i].axis('off')
        if classes:
            ax[i].set_title(classes[res[0].argmax().item()][1])
    plt.show()

def plot_convolution(data_train, t, title=''):
    with torch.no_grad():
        c = nn.Conv2d(kernel_size=(3,3),out_channels=1,in_channels=1)
        c.weight.copy_(t)
        fig, ax = plt.subplots(2,6,figsize=(8,3))
        fig.suptitle(title,fontsize=16)
        for i in range(5):
            im = data_train[i][0]
            ax[0][i].imshow(im[0])
            ax[1][i].imshow(c(im.unsqueeze(0))[0][0])
            ax[0][i].axis('off')
            ax[1][i].axis('off')
        ax[0,5].imshow(t)
        ax[0,5].axis('off')
        ax[1,5].axis('off')
        #plt.tight_layout()
        plt.show()