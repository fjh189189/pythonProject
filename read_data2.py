from utils import *
root_dir = "hymenoptera_data/train"
ants_label_dir = "ants_image"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir,ants_label_dir)
bees_dataset = MyData(root_dir,bees_label_dir)