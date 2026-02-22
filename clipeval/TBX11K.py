import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as transF
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision

Labels = {
    "Tuberculosis": [1, 0],
    "Non-Tuberculosis": [0, 1],
}


class TBCXRDataset(Dataset):
    def __init__(self, root_dir, gt, transform) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.gt_file = gt

        with open(os.path.join(root_dir, 'lists', gt), "r") as f:
            image_list = f.readlines()
        labels_str = ['Tuberculosis' if f.split('/')[0] == 'tb' else 'Non-Tuberculosis' for f in image_list]

        self.all_imgs = np.asarray([f.strip() for f in image_list])
        self.gr = np.array([Labels[label] for label in labels_str])

    def __len__(self):
        return len(self.gr)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, "imgs", self.all_imgs[index])
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

    dataset = TBCXRDataset(root_dir='/mnt/c/chong/data/Xray/TBX11K', gt='TBX11K_val.txt', transform=transform)  # 1800 test samples

    train_loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False, num_workers=0)
    for i, data in enumerate(train_loader):
        batch_size = len(data)

