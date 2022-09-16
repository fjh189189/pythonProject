import argparse
import torch
import torchvision
import importlib
from torch.utils.data import DataLoader
from utils import MyDateset
from utils.tools import print_info,print_train_val_info
from tqdm import  tqdm

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default='config/custom', help='config path')
    return parser.parse_args()
opts = parse_opt()
cfg_name = '.'.join(opts.configs.split('/'))
module = importlib.import_module(cfg_name)
cfg = module.cfg

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# train_set = MyDateset(train=True)
# val_set = MyDateset(train=False)
if cfg['dataset_type'] == 'CIFAR10':
    train_set = torchvision.datasets.CIFAR10("../data", train=True, transform=cfg['trans'],
                                             download=True)
    val_set = torchvision.datasets.CIFAR10("../data", train=False, transform=cfg['trans'],
                                           download=True)
elif cfg['dataset_type'] == 'MyDataSet':
    train_set = MyDateset(cfg,True)
    val_set = MyDateset(cfg,False)

train_loader = DataLoader(train_set,batch_size=cfg['batch_size'],shuffle=True,drop_last=True)
val_loader = DataLoader(val_set,batch_size=cfg['batch_size'],shuffle=True,drop_last=True)

model = cfg['model'].to(device)
lose_function = cfg['lose_function']
if cfg['optim'] == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(),lr=cfg['learning_rate'])
elif cfg['optim'] == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
print_info(cfg)
epoch = 0
epochs = cfg['epochs']
for i in range(epochs):

    train_lose = 0.0
    val_lose = 0.0
    train_total_num = 0
    val_total_num = 0
    train_true_num = 0
    val_true_num = 0
    model.train()
    with tqdm(total=len(train_loader), desc=f'Train: Epoch {epoch + 1}/{epochs}', postfix=dict,
              mininterval=0.3) as pbar:
        for idx,data in enumerate(train_loader):
            img, target = data
            img = img.to(device)
            train_total_num += img.shape[0]
            target = target.to(device)
            output = model(img.to(device))
            # print(output.shape)
            optimizer.zero_grad()
            lose = lose_function(output,target.to(device))
            train_lose += lose.item()
            train_true_num += (output.argmax(1) == target).sum().cpu().item()
            lose.backward()
            optimizer.step()
            # print(f'训练第{idx}iter轮损失值为{train_lose}。。。。。。。。。。。。。。')
    train_acc = round((train_true_num / train_total_num)*100,2)
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f'Valid: Epoch {epoch + 1}/{epochs}', postfix=dict,
                  mininterval=0.3) as pbar:
            for idx,data in enumerate(val_loader):
                img, target = data
                img = img.to(device)
                target = target.to(device)
                val_total_num += img.shape[0]
                output = model(img)
                lose = lose_function(output, target)
                val_lose += lose.item()
                val_true_num += (output.argmax(1) == target).sum().cpu().item()
        val_acc = round((val_true_num / val_total_num)*100,2)
        # print(f'验证第{i}轮损失值为{val_lose}。。。。。。。。。。。。。。')
        # print(f'验证第{i}轮准确率为{accuracy_rate}。。。。。。。。。。。。。。')
    epoch+=1

    print_train_val_info(train_lose,val_lose,train_acc,val_acc)



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