import argparse
import torch
import torchvision
import importlib
from torch.utils.data import DataLoader

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default='config/cifar10', help='config path')
    return parser.parse_args()
opts = parse_opt()
cfg_name = '.'.join(opts.configs.split('/'))
module = importlib.import_module(cfg_name)
cfg = module.cfg


# train_set = MyDateset(train=True)
# val_set = MyDateset(train=False)
train_set = torchvision.datasets.CIFAR10("../data", train=True, transform=cfg['trans'],
                                       download=False)
val_set = torchvision.datasets.CIFAR10("../data", train=False, transform=cfg['trans'],
                                       download=False)

train_loader = DataLoader(train_set,batch_size=cfg['batch_size'],shuffle=True,drop_last=True)
val_loader = DataLoader(val_set,batch_size=cfg['batch_size'],shuffle=True,drop_last=True)

model = cfg['model']
lose_function = cfg['lose_function']
if cfg['optim'] == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(),lr=cfg['learning_rate'])
elif cfg['optim'] == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])

for i in range(cfg['epochs']):
    train_lose = 0.0
    model.train()
    for idx,data in enumerate(train_loader):
        img, target = data
        output = model(img)
        # print(output.shape)
        optimizer.zero_grad()
        lose = lose_function(output,target)
        train_lose += lose.item()
        lose.backward()
        optimizer.step()
        print(f'训练第{idx}iter轮损失值为{train_lose}。。。。。。。。。。。。。。')
    print(f'训练第{i}轮损失值为{train_lose}。。。。。。。。。。。。。。')
    model.eval()
    val_lose = 0.0
    total_num = 0
    true_num = 0
    with torch.no_grad:
        for idx,data in enumerate(val_loader):
            img, target = data
            total_num += img.shape[0]
            output = model(img)
            lose = lose_function(output, target)
            val_lose += lose.item()
            true_num += torch.sum(torch.argmax(output,1) == target).cpu().item()
        accuracy_rate = round((true_num / total_num),2)
        print(f'验证第{i}轮损失值为{val_lose}。。。。。。。。。。。。。。')
        print(f'验证第{i}轮准确率为{accuracy_rate}。。。。。。。。。。。。。。')





# for data in train_loader:
#     img,target = data
#     print(img.shape)
#     print(target.shape)
#     print(model(img).shape)
#     lose = lose_function(model(img), target)
#     print(lose)
#     break
torch.save(model,'model.pth')
# model = torch.load('model.pth')