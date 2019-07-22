import numpy as np 
import torch 
import helper
import matplotlib.pyplot as plt 

from torchvision import datasets, transforms

def activation(x):
    return 1/(1+torch.exp(-x))

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)

# Define a transfrom to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)), ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

# Flatten the input images
inputs = images.view(images.shape[0], -1)

# Create parameters
w1 = torch.randn(784, 256)
b1 = torch.randn(256)

w2 = torch.randn(256, 10)
b2 = torch.randn(10)

h1 = torch.mm(inputs, w1) + b1
h = activation(h1)

output = torch.mm(h, w2) + b2

probabilities = softmax(output)

# Does it have the rigth shape? Should it be (6, 4, 10)
print(probabilities.shape)
# Does it sum up to 1
print(probabilities.sum(dim=1))

plt.imshow(images[1].numpy().squeeze(), cmap=('Greys_r'))