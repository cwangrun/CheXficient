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


Labels = ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Clavicle fracture',
          'Consolidation', 'Edema', 'Emphysema', 'Enlarged PA', 'ILD', 'Infiltration', 'Lung Opacity', 'Lung cavity',
          'Lung cyst', 'Mediastinal shift', 'Nodule/Mass', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
          'Pulmonary fibrosis', 'Rib fracture', 'COPD', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other lesion',
          'Other disease', 'No finding']


class VinDrCXRDataset(Dataset):
    def __init__(self, root_dir, gt, transform) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.gt_file = gt

        df_csv = pd.read_csv(os.path.join(root_dir, gt))

        self.all_imgs = np.asarray([x for x in df_csv['image_id']])
        self.gr = np.array([df_csv[label] for label in Labels]).transpose()

    def __len__(self):
        return len(self.gr)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, "test_png", self.all_imgs[index] + '.png')
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

    dataset = VinDrCXRDataset(root_dir='/mnt/c/chong/data/vindr-cxr', gt='image_labels_test.csv', transform=transform)  # 3000 test samples

    train_loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False, num_workers=0)
    for i, data in enumerate(train_loader):
        batch_size = len(data)
