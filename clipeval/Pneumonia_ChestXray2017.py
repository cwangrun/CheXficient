import glob
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision


Labels = {
    "PNEUMONIA": [1, 0],
    "NORMAL": [0, 1],
}


class Pneumonia_Xray2017(Dataset):
    def __init__(self, root_dir, transform) -> None:
        self.root_dir = root_dir
        self.transform = transform

        img_list = glob.glob(os.path.join(root_dir, '*', '*.*'))
        gr_str = [path.split('/')[-2] for path in img_list]
        self.all_imgs = np.asarray(img_list)
        self.gr = np.asarray([Labels[g] for g in gr_str])

    def __len__(self):
        return len(self.gr)

    def __getitem__(self, index):
        img_path = self.all_imgs[index]
        img = Image.open(img_path).convert("RGB")
        data = self.transform(img)
        target = torch.tensor(self.gr[index]).long()
        return data, target, img_path
        # return data, target, self.gr_str[index]


if __name__ == '__main__':

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    dataset = Pneumonia_Xray2017(root_dir='/mnt/c/chong/data/Xray/ChestXRay2017/chest_xray/test/', transform=transform)   # 624 test samples

    train_loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False, num_workers=10)
    for i, data in enumerate(train_loader):
        batch_size = len(data)

