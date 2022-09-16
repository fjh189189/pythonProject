from torchvision import transforms
cfg = {
    "label2idx":{"ants":0,"bees":1},
    "root_path":"hymenoptera_data",
    "trans":transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor()])
}