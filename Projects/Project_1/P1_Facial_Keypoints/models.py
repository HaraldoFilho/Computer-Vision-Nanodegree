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
        # self.conv1 = nn.Conv2d(1, 32, 5)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers,
        # and other layers (such as dropout or batch normalization) to avoid overfitting

        # 1 input image channel, 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5) # input = 224x224x1, output = 220x220x32

        # maxpool that uses a square window of kernel_size=2, stride=2
        self.pool1 = nn.MaxPool2d(4, 4) # output = input/4

        # 32 input image channels, 64 output channels/feature maps, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(32, 64, 6) # input = 55x55x32, output = 50x50x64

        # maxpool that uses a square window of kernel_size=2, stride=2
        self.pool2 = nn.MaxPool2d(2, 2) # output = input/2

        # 64 input image channels, 128 output channels/feature maps, 5x5 square convolution kernel
        self.conv3 = nn.Conv2d(64, 128, 6) # input = 25x25x64, output = 20x20x128

        # maxpool that uses a square window of kernel_size=2, stride=2
        self.pool3 = nn.MaxPool2d(1, 1) # output = input

        # fully connected layer with 2*136 output values
        self.fc1 = nn.Linear(20*20*128, 64)

        # dropout with p=0.4
        self.drop_fc1 = nn.Dropout(p=0.4)

        # fully connected layer with 136 output values (68 coordinates/keypoints)
        self.fc2 = nn.Linear(64, 136)


    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        # one conv/relu + pool layer
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)

        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.drop_fc1(x)
        x = self.fc2(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
