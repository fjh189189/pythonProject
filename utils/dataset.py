import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from cfg import cfg
class MyDateset(Dataset):

    def __init__(self,train=True):
        # json_file = 'cfg.json'
        # with open(json_file, 'r', encoding='utf-8') as f:
        #     j = json.load(f)
        root_path = cfg['root_path']
        label2idx = cfg['label2idx']
        trans = cfg['trans']
        if train:
            dataset_path = os.path.join(root_path,'train')
        else:
            dataset_path = os.path.join(root_path, 'val')
        label_list = os.listdir(dataset_path)
        # idx_list = [label2idx[label] for label in label_list]
        self.img_list = []
        self.target_list = []
        for label in label_list:
            img_path = os.path.join(dataset_path, label)
            for img_name in  os.listdir(img_path):
                single_img_path = os.path.join(img_path,img_name)
                #shape[h*w*c]
                # img_array = np.array(Image.open(single_img_path)).transpose(2,0,1)
                #shape[c*h*w]
                img_tensor = trans(Image.open(single_img_path))
                # img_tensor = torch.tensor(img_array/255)

                self.img_list.append(img_tensor)
                self.target_list.append(torch.tensor(label2idx[label]))
    def __getitem__(self, idx):
        return self.img_list[idx],self.target_list[idx]

    def __len__(self):
        return len(self.img_list)