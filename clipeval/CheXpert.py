import numpy as np
import torch
import random, os
from PIL import Image, ImageFile
from torchvision import datasets as t_datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pathlib import Path
from typing import Dict, List
from torch.utils.data import default_collate
import ast
from constants import CHEXPERT_CLASS_PROMPTS
from transformers import AutoTokenizer, AutoModel


class CheXpert5x200Dataset(Dataset):
    def __init__(
        self,  root_dir, gt, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.label_list = list(CHEXPERT_CLASS_PROMPTS.keys())

        self.idx2label = {idx: self.label_list[idx] for idx in range(len(self.label_list))}
        self.label2idx = {v: k for k, v in self.idx2label.items()}

        self.df_csv = pd.read_csv(os.path.join(root_dir, gt))

    def __len__(self):
        return len(self.df_csv)

    def __getitem__(self, index):
        image_path = self.df_csv["Path"][index]

        if image_path.startswith("["):
            image_path = ast.literal_eval(image_path)[0]  # not random sampling

        img = Image.open(str(self.root_dir + '/../' + image_path)).convert("RGB")

        image = self.transform(img)

        # text = self.df["Report Impression"][index]
        sample = {"images": image}

        for label_candidate in self.label_list:
            if self.df_csv[label_candidate][index] == 1.0:
                label = label_candidate
        label_idx = self.label2idx[label]

        sample["label_names"] = label
        sample["label_indices"] = label_idx

        return sample


CheXpert_labels = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema',
                   'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                   'Pleural Other', 'Fracture', 'Support Devices', 'No Finding']

class CheXpertTestDataset(Dataset):
    def __init__(self, root_dir, gt, transform) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.gt_file = gt

        cxr_dir = Path(os.path.join(root_dir, 'CheXpert/test'))
        cxr_paths = list(cxr_dir.rglob("*.jpg"))
        cxr_paths = list(filter(lambda x: "view1" in str(x), cxr_paths))  # filter only first frontal views
        self.cxr_paths = sorted(cxr_paths)  # sort to align with groundtruth

        df_csv = pd.read_csv(os.path.join(root_dir, gt))
        self.gr = np.array([df_csv[label] for label in CheXpert_labels]).transpose()

    def __len__(self):
        return len(self.gr)

    def __getitem__(self, index):
        img_path = os.path.join(self.cxr_paths[index])
        img = Image.open(img_path).convert("RGB")
        data = self.transform(img)
        target = torch.tensor(self.gr[index]).long()
        return data, target, img_path
        # return data, target, self.gr_str[index]


class CheXpertRetrieveDataset(Dataset):
    def __init__(self, root_dir, gt, transform, max_bert_length=256) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.gt_file = gt
        self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', cache_dir='../../huggingface/tokenizers')
        self.max_bert_length = max_bert_length

        df_csv_gt = pd.read_csv(os.path.join(root_dir, "chexpert_5x200.csv"), low_memory=False)
        gt_paths = df_csv_gt['Path'].apply(lambda path: path.split('CheXpert-v1.0/')[1]).tolist()
        df_csv = pd.read_csv(os.path.join(root_dir, 'df_chexpert_plus_240401.csv'))   # 1000
        mask = df_csv["path_to_image"].apply(lambda path: path in gt_paths)
        df_csv_new = df_csv[mask]  # 998

        df_csv_new.dropna(subset=["section_findings", "section_impression"], how="all", inplace=True)
        df_csv_new[["section_findings"]] = df_csv_new[["section_findings"]].fillna(" ")
        df_csv_new[["section_impression"]] = df_csv_new[["section_impression"]].fillna(" ")

        self.df = df_csv_new
        self.all_imgs = np.asarray([x for x in self.df['path_to_image']])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.all_imgs[index])
        img = Image.open(img_path).convert("RGB")
        data = self.transform(img)

        row = self.df.iloc[index]
        captions = ""
        captions += row["section_impression"]
        captions += " "
        captions += row["section_findings"]

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

    # dataset = CheXpertTestDataset(root_dir='/mnt/c/chong/data/CheXpert-v1.0/chexlocalize/', gt='groundtruth.csv', transform=transform)  # 500 test samples
    dataset = CheXpert5x200Dataset(root_dir='/mnt/c/chong/data/CheXpert-v1.0/', gt='chexpert_5x200.csv', transform=transform)  # 500 test samples

    train_loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False, num_workers=0)
    for i, data in enumerate(train_loader):
        batch_size = len(data)
