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
from transformers import AutoTokenizer, AutoModel


Labels = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema',
          'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
          'Pleural Other', 'Fracture', 'Support Devices', 'No Finding']


class MIMICCXRDataset(Dataset):
    def __init__(self, root_dir, gt, transform) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.gt_file = gt

        df_csv = pd.read_csv(os.path.join(root_dir, '2.0.0/yulequan', gt))

        df_csv = df_csv[df_csv["ViewPosition"].isin(["AP", 'PA'])]  # only consider frontal view

        self.all_imgs = np.asarray([x for x in df_csv['Path']])
        self.gr = np.array([df_csv[label].fillna(0) for label in Labels]).transpose()  # no annotation as 0, uncertainty is -1

    def __len__(self):
        return len(self.gr)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.all_imgs[index])
        img = Image.open(img_path).convert("RGB")
        data = self.transform(img)
        target = torch.tensor(self.gr[index]).long()
        return data, target, img_path
        # return data, target, self.gr_str[index]


class MIMICCXRRetrieveDataset(Dataset):
    def __init__(self, root_dir, gt, transform, max_bert_length=256) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.gt_file = gt
        self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', cache_dir='../../huggingface/tokenizers')
        self.max_bert_length = max_bert_length

        df_csv = pd.read_csv(os.path.join(root_dir, '2.0.0/yulequan', gt))
        df_csv = df_csv[df_csv["split"] == 'test']

        df_csv = df_csv[df_csv["ViewPosition"].isin(["AP", 'PA'])]  # only consider frontal view

        self.all_imgs = np.asarray([x for x in df_csv['Path']])
        self.df = df_csv

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.all_imgs[index])
        img = Image.open(img_path).convert("RGB")
        data = self.transform(img)

        row = self.df.iloc[index]
        captions = ""
        captions += row["impression"]
        captions += " "
        captions += row["findings"]

        # use space instead of newline
        captions_raw = captions.replace("\n", " ")

        # # padding="max_length" 保证 batch 内长度一致
        # text_tokens = self.tokenizer(
        #     captions_raw, padding="max_length", truncation=True, return_tensors="pt", max_length=self.max_bert_length
        # )

        # # squeeze 掉 batch 维度: [1, L] -> [L]
        # text_tokens = {k: v.squeeze(0) for k, v in text_tokens.items()}

        # return data, text_tokens, captions_raw, img_path
        return data, captions_raw, img_path


if __name__ == '__main__':

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    dataset = MIMICCXRDataset(root_dir='/mnt/c/chong/data/mimic-cxr-jpg', gt='test.csv', transform=transform)  # 3082 test samples

    train_loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False, num_workers=0)
    for i, data in enumerate(train_loader):
        batch_size = len(data)
