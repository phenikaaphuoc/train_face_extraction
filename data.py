import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import glob
import os
from PIL import Image
import random
from torchvision import transforms
from  tqdm import tqdm
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes()
        self.num_class = len(glob.glob(os.path.join(root_dir,"*")))

    def _find_classes(self):
        classes = [d.name for d in os.scandir(self.root_dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        return classes, class_to_idx

    def __len__(self):
        return sum(len(files) for _, _, files in os.walk(self.root_dir))

    def __getitem__(self, idx):
        class_folders = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        selected_class = class_folders[idx % len(class_folders)]
        class_path = os.path.join(self.root_dir, selected_class)
        img_name = random.choice(os.listdir(class_path))
        img_path = os.path.join(class_path, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.class_to_idx[selected_class])

        return image, torch.tensor(nn.functional.one_hot(label,self.num_class),dtype = torch.float32)


def get_dataloader(opt,train = True):
    if not train:
        transform = transforms.Compose([
            transforms.Resize((opt['image_size'], opt['image_size'])),  # Resize the image to the desired size
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((opt['image_size'], opt['image_size'])),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
        ])
    dataset = CustomDataset(root_dir=opt["data_dir"], transform=transform)
    dataloader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)
    return dataloader

