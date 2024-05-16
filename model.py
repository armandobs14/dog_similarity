import torch.nn as nn


# create the Siamese Neural Network
class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        # self.cnn1 = nn.Sequential(
        #     nn.Conv2d(3, 256, kernel_size=11,stride=4),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(3, stride=2),

        #     nn.Conv2d(256, 256, kernel_size=5, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, stride=2),

        #     nn.Conv2d(256, 384, kernel_size=3,stride=1),
        #     nn.ReLU(inplace=True)
        # )

        self.cnn1 = nn.Conv2d(3, 256, kernel_size=11, stride=4)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.cnn2 = nn.Conv2d(256, 256, kernel_size=5, stride=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.cnn3 = nn.Conv2d(256, 384, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(46464, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)
        # Setting up the Fully Connected Layers
        # self.fc1 = nn.Sequential(
        #     nn.Linear(384, 1024),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(1024, 32*46464),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(32*46464,1)
        # )

    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        # output = self.cnn1(x)
        # print(output.view(output.size()[0], -1).shape)
        # output = output.view(output.size()[0], -1)
        # output = self.fc1(output)
        # print(x.shape)
        output = self.cnn1(x)
        # print(output.shape)
        output = self.relu(output)
        # print(output.shape)
        output = self.maxpool1(output)
        # print(output.shape)
        output = self.cnn2(output)
        # print(output.shape)
        output = self.relu(output)
        # print(output.shape)
        output = self.maxpool2(output)
        # print(output.shape)
        output = self.cnn3(output)
        output = self.relu(output)
        # print(output.shape)
        output = output.view(output.size()[0], -1)
        # print(output.shape)
        output = self.fc1(output)
        # print(output.shape)
        output = self.fc2(output)
        # print(output.shape)
        output = self.fc3(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2
