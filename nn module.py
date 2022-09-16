import torchvision
from torchvision import models
from torch import conv2d, nn
from torch.nn import MaxPool2d, ReLU
from torch.utils.data import DataLoader
import torch
from utils import CIFAR10Model
import torch.nn.functional as F

3
# class Model(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self,input):
#         output = input+1
#         return output
# model=Model()
# x=torch.tensor(1.0)
# output=model(x)
# print(output)


input = torch.tensor([[1, -0.5],
                      [-1,3]])
# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype = torch.float32)

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)
# kernel = torch.reshape(kernel, (1, 1, 3, 3))
#
# output = F.conv2d(input, kernel, stride=2, padding=1)
# print(output)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3, ceil_mode=True)
        self.relu = ReLU()

    def forward(self, input):
        output = self.relu(input)
        return output

model = Model()
output = model(input)
print(output)




dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)
#
# class Model(nn.Module):
#
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = conv2d(in_channels=3, out_channel=6, kernel_size=3, stride=1, padding=0)
#
#         def forward(self, x):
#             x = self.conv1(x)
#             return x
#
# model=Model()
# print(model)
mymodel = CIFAR10Model(10)
for data in dataloader:
    imgs, targets = data
    output = mymodel(imgs)
    print(output.shape)

    # print(imgs.shape)
    # print(output.shape)
