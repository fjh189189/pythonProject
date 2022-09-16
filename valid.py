import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

idx2label = {0:"ants",1:"bees"}
model = torch.load('model.pth',map_location=torch.device('cpu'))
val_set = torchvision.datasets.CIFAR10("../data", train=False, transform=transforms.ToTensor(),
                                           download=False)
val_loader = DataLoader(val_set,batch_size=10,shuffle=True,drop_last=True)
for img,target in val_loader:
    output = model(img)
    pred = output.argmax(1)
    # pred = idx2label[output.argmax(1).item()]
    true = target
    print(f'预测的类别为{pred}')
    print(f'真实的类别为{true}')
    break
