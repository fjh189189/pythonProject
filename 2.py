import json
import os.path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MyData(Dataset):

    def __init__(self,train=True):
        json_file='cfg.json'
        with open(json_file,'r',encoding='utf-8') as f:
            j=json.load(f)
            root_path=j['root_path']
            label2index=j['label2idx']
        self.img_list=[]
        self.label_list=[]
        if train:
            set_path=os.path.join(root_path,"train");
        else:
            set_path = os.path.join(root_path, "val");
        label_list=os.listdir(set_path)
        for label in label_list:
            img_path=os.path.join(set_path,label)
            img_list=os.listdir(img_path)
            for img in img_list:
               total_path = os.path.join(img_path, img)
               image_tensor=torch.tensor(np.array(Image.open(total_path)).transpose(2,0,1))/255
               label_tensor=torch.tensor(label2index[label])
               self.img_list.append(image_tensor)
               self.label_list.append(label_tensor)
    def __getitem__(self, item):
        return self.img_list[item],self.label_list[item]

    def __len__(self):
        return len(self.label_list)


# root_path="hymenoptera_data"
# label2index={"ants_image":0,"bees":1}
mydata=MyData()
print(len(mydata))


