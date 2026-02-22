import glob
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
from PIL import Image
import numpy as np

import pydicom
import matplotlib.pyplot as plt


class SIIMPneumothoraxCXRDataset(Dataset):
    def __init__(self, root_dir, gt, transform) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.gt_file = gt

        all_imgs = np.array(glob.glob(os.path.join(root_dir, 'png-images-test-512', '*.png')))
        df_csv = pd.read_csv(os.path.join(root_dir, gt))
        mask_exist = np.array([True if os.path.splitext(os.path.basename(p))[0] in list(df_csv["ImageId"]) else False for p in all_imgs])
        self.all_imgs = all_imgs[mask_exist]
        ground_truth = [df_csv.loc[df_csv['ImageId'] == os.path.splitext(os.path.basename(img_id))[0], ' EncodedPixels'].values[0].strip() for img_id in self.all_imgs]
        temp = set(ground_truth)   # for check
        self.gr = np.array([[0, 1] if g == '-1' else [1, 0] for g in ground_truth])   # {'Pneumothorax': [1, 0], 'NORMAL': [0, 1]}

    def __len__(self):
        return len(self.gr)

    def __getitem__(self, index):
        img_path = os.path.join(self.all_imgs[index])
        img = Image.open(img_path).convert("RGB")
        data = self.transform(img)
        target = torch.tensor(self.gr[index]).long()
        return data, target, img_path
        # return data, target, self.gr_str[index]


if __name__ == '__main__':

    def resize_image_keep_aspect(src_path, dst_path, min_side=512, quality=100):
        try:
            img = Image.open(src_path).convert("RGB")
            w, h = img.size
            scale = min_side / min(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            img.save(dst_path, quality=quality)
        except Exception as e:
            print(f"❌ Resize failed for {src_path}: {e}")

    def dicom_to_png(dcm_path, png_path, min_side=512):
        try:
            ds = pydicom.dcmread(dcm_path)
            pixel_array = ds.pixel_array
            # print(pixel_array.shape)

            w, h = pixel_array.shape[0], pixel_array.shape[1]
            scale = min_side / min(w, h)
            new_w, new_h = int(w * scale), int(h * scale)

            image = Image.fromarray(pixel_array)
            image = image.resize((new_w, new_h), Image.LANCZOS)

            image = image.convert('L')  # 灰度图
            image.save(png_path)
        except Exception as e:
            print(f"Error processing {dcm_path}: {e}")


    def convert_all_dicoms_to_png(input_dir, output_dir):
        nnn = 0
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith('.dcm'):
                    dcm_path = os.path.join(root, file)

                    # 构造保存路径（保持相对路径结构）
                    relative_path = os.path.relpath(root, input_dir)
                    # png_dir = os.path.join(output_dir, relative_path)
                    png_dir = output_dir
                    os.makedirs(png_dir, exist_ok=True)

                    png_filename = os.path.splitext(file)[0] + '.png'
                    png_path = os.path.join(png_dir, png_filename)

                    dicom_to_png(dcm_path, png_path)
                    nnn = nnn + 1
        print('number of images: ', nnn)


    # # === 修改这两个路径为你的实际路径 ===
    # input_dicom_dir = "/mnt/c/chong/data/Xray/SIIM ACR Pneumothorax Segmentation Data/pneumothorax/dicom-images-train/"   # 原始DICOM目录  number of images:  1377    1024 resolution
    # output_png_dir = "/mnt/c/chong/data/Xray/SIIM ACR Pneumothorax Segmentation Data/pneumothorax/png-images-train-512/"  # PNG保存目录
    # convert_all_dicoms_to_png(input_dicom_dir, output_png_dir)

    # csvpath = "/mnt/c/chong/data/Xray/SIIM ACR Pneumothorax Segmentation Data/pneumothorax/train-rle_zhihong.csv"
    # df_csv = pd.read_csv(csvpath, low_memory=False)   # 12954
    # paths = list(df_csv['ImageId'])

    # csvpath_ori = "/mnt/c/chong/data/Xray/SIIM ACR Pneumothorax Segmentation Data/pneumothorax/train-rle.csv"
    # df_csv_ori = pd.read_csv(csvpath_ori, low_memory=False)   # 11582
    # paths_ori = list(df_csv_ori['ImageId'])

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    dataset = SIIMPneumothoraxCXRDataset(root_dir='/mnt/c/chong/data/Xray/SIIM-ACR-Pneumothorax-Segmentation-Data/pneumothorax', gt='train-rle_zhihong.csv', transform=transform)  # 1372 test samples, 10675 train samples

    train_loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False, num_workers=10)
    for i, data in enumerate(train_loader):
        batch_size = len(data)
        print('data size: ', batch_size)

