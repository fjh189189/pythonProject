from torchvision import transforms
from utils import *

cfg = {
    'dataset_type':'MyDataSet',
    "label2idx":{"ants":0,"bees":1},
    "root_path":"hymenoptera_data",
    'epochs':1,
    'model':AlexNet(10),
    'lose_function':nn.CrossEntropyLoss(),
    # 'cls_num' : 10,
    'learning_rate':0.01,
    'batch_size':4,
    "trans":transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor()]),
    'optim':'SGD'
}