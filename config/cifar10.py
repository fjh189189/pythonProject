from torchvision import transforms
from utils import *

cfg = {
    'epochs':50,
    'model':CIFAR10Model(10),
    'lose_function':nn.CrossEntropyLoss(),
    # 'cls_num' : 10,
    'learning_rate':0.01,
    'batch_size':256,
    "trans":transforms.Compose([transforms.ToTensor()]),
    'optim':'SGD'
}