## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5) # input = 224x224x1, output = 220x220x32

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers,
        # and other layers (such as dropout or batch normalization) to avoid overfitting

        # maxpool that uses a square window of kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2) # input = 220x220x32, output = 110x110x32

        # fully connected layer with 136 output values (68 coordinates/keypoints)
        self.fc1 = nn.Linear(110*110*32, 136)


    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        # one conv/relu + pool layer
        x = self.pool(F.relu(self.conv1(x)))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)

        # one linear layer
        x = F.relu(self.fc1(x))

        # a modified x, having gone through all the layers of your model, should be returned
        return x
