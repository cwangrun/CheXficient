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
import random


Labels = {
    "No Finding": 14,
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Effusion": 2,
    "Infiltration": 3,
    "Mass": 4,
    "Nodule": 5,
    "Pneumonia": 6,
    "Pneumothorax": 7,
    "Consolidation": 8,
    "Edema": 9,
    "Emphysema": 10,
    "Fibrosis": 11,
    "Pleural_Thickening": 12,
    "Hernia": 13,
}


mlb = MultiLabelBinarizer(classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])


class ChestXray14Dataset(Dataset):
    def __init__(self, root_dir, gt, transform) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.gt_file = gt

        gr_path = os.path.join(root_dir, "Data_Entry_2017.csv")
        gr = pd.read_csv(gr_path, index_col=0)
        gr = gr.to_dict()["Finding Labels"]

        img_list = os.path.join(root_dir, gt)
        with open(img_list) as f:
            all_names = f.read().splitlines()
            # all_names = all_names[0:287]
        self.all_imgs = np.asarray([x for x in all_names])
        self.gr_str = np.asarray([gr[i] for i in self.all_imgs])
        self.gr = np.zeros((self.gr_str.shape[0], len(Labels)))
        for idx, i in enumerate(self.gr_str):
            target = i.split("|")
            binary_result = mlb.fit_transform([[Labels[i] for i in target]]).squeeze()
            self.gr[idx] = binary_result

    def __len__(self):
        return len(self.gr)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, "data", self.all_imgs[index])
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

    dataset = ChestXray14Dataset(root_dir='/mnt/c/chong/data/chestxray14', gt='test_list.txt', transform=transform)   # 25596 test samples

    train_loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False, num_workers=0)
    for i, data in enumerate(train_loader):
        batch_size = len(data)
