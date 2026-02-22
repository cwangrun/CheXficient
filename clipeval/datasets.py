# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import json
import os
import pickle
import zipfile

import numpy as np
import torch
import random

from PIL import Image, ImageFile
from torchvision import datasets as t_datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

from constants import *

from typing import Dict, List
from torch.utils.data import default_collate
from .CheXpert import CheXpertTestDataset, CheXpert5x200Dataset, CheXpertRetrieveDataset
from .MIMIC_CXR import MIMICCXRDataset, MIMICCXRRetrieveDataset
from .NIHChestXray14 import ChestXray14Dataset
from .Pneumonia_ChestXray2017 import Pneumonia_Xray2017
from .SIIM_Pneumothorax import SIIMPneumothoraxCXRDataset
from .TBX11K import TBCXRDataset
from .VinDr_CXR import VinDrCXRDataset
from .VinDr_PCXR import VinDrPCXRDataset
from .ReXGraident import ReXGraidentRetrieveDataset   


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_path_2_sent_mapping_merge(df, findings_name, impression_name, path_name):
    path2sent = {}
    # iterrows is not faster than itertuples ...  but it is ok
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        # pick impression, findings, last_paragraph
        captions = ""
        captions += row[findings_name]
        captions += " "
        captions += row[impression_name]

        # use space instead of newline
        captions_raw = captions.replace("\n", " ")

        path2sent[row[path_name]] = captions_raw

    return path2sent


def create_path_2_sent_mapping_standalone(df, report_name, path_name):
    path2sent = {}
    # iterrows is not faster than itertuples ...  but it is ok
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        # pick impression, findings, last_paragraph
        captions = ""
        captions += row[report_name]

        # use space instead of newline
        captions_raw = captions.replace("\n", " ")

        path2sent[row[path_name]] = captions_raw

    return path2sent


class MultimodalPretrainingDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", transform=None, example_ids=None, max_bert_length=256, tokenizer=None):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_bert_length = max_bert_length
        self.transform = transform

        ########################################## chexpert plus ######################################################
        if not os.path.exists(CHEXPERT_DATA_DIR):
            raise RuntimeError(f"{CHEXPERT_DATA_DIR} does not exist!")
        df_chexpert = pd.read_csv(CHEXPERT_TRAIN_CSV)  # 222116
        # df_chexpert = df_chexpert[df_chexpert["frontal_lateral"].isin(["Frontal"])]  # 189774
        df_chexpert['path_to_image'] = df_chexpert['path_to_image'].apply(lambda path: os.path.join(CHEXPERT_DATA_DIR, path))
        path2sent_chexpert = create_path_2_sent_mapping_merge(df_chexpert, 'section_findings', 'section_impression', 'path_to_image')
        # filter studies to use for current split
        filenames_chexpert = []
        for row in df_chexpert.itertuples():
            cur_split = getattr(row, 'split')
            path = getattr(row, 'path_to_image')
            if cur_split == split and path in path2sent_chexpert:
                filenames_chexpert.append(path)
        ########################################## mimic ##############################################################
        if not os.path.exists(MIMIC_CXR_DATA_DIR):
            raise RuntimeError(f"{MIMIC_CXR_DATA_DIR} does not exist!")
        df_mimic = pd.read_csv(MIMIC_CXR_MASTER_CSV)  # 362195
        # df_mimic = df_mimic[df_mimic["ViewPosition"].isin(["AP", 'AP AXIAL', 'AP LLD', 'AP RLD', 'PA', 'PA LLD', 'PA RLD'])]  # 232431
        df_mimic['Path'] = df_mimic['Path'].apply(lambda path: os.path.join(MIMIC_CXR_DATA_DIR, "/".join(path.split("/")[1:])))
        path2sent_mimic = create_path_2_sent_mapping_merge(df_mimic, 'findings', 'impression', 'Path')
        # filter studies to use for current split
        filenames_mimic = []
        for row in df_mimic.itertuples():
            cur_split = getattr(row, 'split')
            path = getattr(row, 'Path')
            if cur_split == split and path in path2sent_mimic:
                filenames_mimic.append(path)
        ########################################## PadChest ###########################################################
        if not os.path.exists(PadChest_CXR_DATA_DIR):
            raise RuntimeError(f"{PadChest_CXR_DATA_DIR} does not exist!")
        df_padchest = pd.read_csv(PadChest_CXR_TRAIN_CSV)  # 160672
        # df_padchest = df_padchest[df_padchest["Projection"].isin(['AP', 'AP_horizontal', 'PA', 'COSTAL'])]  # 111125
        df_padchest['ImageID'] = df_padchest['ImageID'].apply(lambda path: os.path.join(PadChest_CXR_DATA_DIR, 'images', path))
        path2sent_padchest = create_path_2_sent_mapping_standalone(df_padchest, 'Report_English', 'ImageID')
        # filter studies to use for current split
        filenames_padchest = []
        for row in df_padchest.itertuples():
            path = getattr(row, 'ImageID')
            if path in path2sent_padchest:
                filenames_padchest.append(path)
        ########################################### open-i ############################################################
        if not os.path.exists(Open_I_CXR_DATA_DIR):
            raise RuntimeError(f"{Open_I_CXR_DATA_DIR} does not exist!")
        df_openi = pd.read_csv(Open_I_CXR_TRAIN_CSV)  # 7424
        # df_openi = df_openi[df_openi["projection"].isin(['Frontal'])]  # 3793
        df_openi['filename'] = df_openi['filename'].apply(lambda path: os.path.join(Open_I_CXR_DATA_DIR, 'NLMCXR_png', path))
        path2sent_openi = create_path_2_sent_mapping_merge(df_openi, 'findings', 'impression', 'filename')
        # filter studies to use for current split
        filenames_openi = []
        for row in df_openi.itertuples():
            path = getattr(row, 'filename')
            if path in path2sent_openi:
                filenames_openi.append(path)
        ########################################### bimcv #############################################################
        if not os.path.exists(BIMCV_CXR_DATA_DIR):
            raise RuntimeError(f"{BIMCV_CXR_DATA_DIR} does not exist!")
        df_bimcv = pd.read_csv(BIMCV_CXR_TRAIN_CSV)  # 65421
        # df_bimcv = df_bimcv[df_bimcv["view"].isin(['ap', 'pa'])]  # 57316
        df_bimcv['file_path'] = df_bimcv['file_path'].apply(lambda path: os.path.join(BIMCV_CXR_DATA_DIR, 'images', path))
        path2sent_bimcv = create_path_2_sent_mapping_standalone(df_bimcv, 'report_english', 'file_path')
        # filter studies to use for current split
        filenames_bimcv = []
        for row in df_bimcv.itertuples():
            path = getattr(row, 'file_path')
            if path in path2sent_bimcv:
                filenames_bimcv.append(path)
        ########################################### CASIA-CXR #########################################################
        if not os.path.exists(CASIA_CXR_DATA_DIR):
            raise RuntimeError(f"{CASIA_CXR_DATA_DIR} does not exist!")
        df_casia = pd.read_csv(CASIA_CXR_TRAIN_CSV)  # 11111
        # all view in this dataset are frontal
        df_casia['ImageDir'] = df_casia['ImageDir'].apply(lambda path: os.path.join(CASIA_CXR_DATA_DIR, path))
        path2sent_casia = create_path_2_sent_mapping_merge(df_casia, 'Findings_Eng', 'Impression_Eng', 'ImageDir')
        # filter studies to use for current split
        filenames_casia = []
        for row in df_casia.itertuples():
            path = getattr(row, 'ImageDir')
            if path in path2sent_casia:
                filenames_casia.append(path)                           
        ########################################## candid #############################################################
        if not os.path.exists(CANDID_CXR_DATA_DIR):
            raise RuntimeError(f"{CANDID_CXR_DATA_DIR} does not exist!")
        df_candid = pd.read_csv(CANDID_CXR_TRAIN_CSV)  # 19609
        # all view in this dataset are frontal
        df_candid['file_path'] = df_candid['file_path'].apply(lambda path: os.path.join(CANDID_CXR_DATA_DIR, path))
        path2sent_candid = create_path_2_sent_mapping_standalone(df_candid, 'report_impression', 'file_path')
        # filter studies to use for current split
        filenames_candid = []
        for row in df_candid.itertuples():
            path = getattr(row, 'file_path')
            if path in path2sent_candid:
                filenames_candid.append(path)
        ########################################## RexGradient-160k ###################################################
        if not os.path.exists(ReXGradient_CXR_DATA_DIR):
            raise RuntimeError(f"{ReXGradient_CXR_DATA_DIR} does not exist!")
        df_ReXGradient = pd.read_csv(ReXGradient_CXR_TRAIN_CSV)  # 238965
        # df_ReXGradient = df_ReXGradient[df_ReXGradient["ImageViewPosition"].isin(['ANTERO_POSTERIOR', 'AP', 'AP AXIAL', 'DECUBITUS', 'ERECT', 'KUB', 'PA', 'PICC LINE', 'POSTERO_ANTERIOR', 'SUPINE', 'UNKNOWN'])]  # 140831
        df_ReXGradient['ImagePath'] = df_ReXGradient['ImagePath'].apply(lambda path: os.path.join(ReXGradient_CXR_DATA_DIR, path))
        path2sent_ReXGradient = create_path_2_sent_mapping_merge(df_ReXGradient, 'Findings', 'Impression', 'ImagePath')
        # filter studies to use for current split
        filenames_ReXGradient = []
        for row in df_ReXGradient.itertuples():
            path = getattr(row, 'ImagePath')
            if path in path2sent_ReXGradient:
                filenames_ReXGradient.append(path)
        ########################################### Brax ##############################################################
        if not os.path.exists(Brax_CXR_DATA_DIR):
            raise RuntimeError(f"{Brax_CXR_DATA_DIR} does not exist!")
        df_brax = pd.read_csv(Brax_CXR_TRAIN_CSV)  # 40967
        # df_brax = df_brax[df_brax["ViewPosition"].isin(['PA', 'AP LLD', 'AP'])]  # 19310
        df_brax['PngPath'] = df_brax['PngPath'].apply(lambda path: os.path.join(Brax_CXR_DATA_DIR, path))
        path2sent_brax = create_path_2_sent_mapping_standalone(df_brax, 'Synthetic_Report', 'PngPath')
        # filter studies to use for current split
        filenames_brax = []
        for row in df_brax.itertuples():
            path = getattr(row, 'PngPath')
            if path in path2sent_brax:
                filenames_brax.append(path)
        ########################################### ChestDR ###########################################################
        if not os.path.exists(ChestDR_CXR_DATA_DIR):
            raise RuntimeError(f"{ChestDR_CXR_DATA_DIR} does not exist!")
        df_ChestDR = pd.read_csv(ChestDR_CXR_TRAIN_CSV)  # 4848
        # all views in this dataset are frontal
        df_ChestDR['img_id'] = df_ChestDR['img_id'].apply(lambda path: os.path.join(ChestDR_CXR_DATA_DIR, 'images', path))
        path2sent_ChestDR = create_path_2_sent_mapping_standalone(df_ChestDR, 'Synthetic_Report', 'img_id')
        # filter studies to use for current split
        filenames_ChestDR = []
        for row in df_ChestDR.itertuples():
            path = getattr(row, 'img_id')
            if path in path2sent_ChestDR:
                filenames_ChestDR.append(path)
        ########################################### NIHChestXray14 ####################################################
        if not os.path.exists(NIHChestXray14_CXR_DATA_DIR):
            raise RuntimeError(f"{NIHChestXray14_CXR_DATA_DIR} does not exist!")
        df_nihchestxray14 = pd.read_csv(NIHChestXray14_CXR_TRAIN_CSV)  # 86524
        # all views in this dataset are frontal
        df_nihchestxray14['Image Index'] = df_nihchestxray14['Image Index'].apply(lambda path: os.path.join(NIHChestXray14_CXR_DATA_DIR, 'data', path))
        path2sent_nihchestxray14 = create_path_2_sent_mapping_standalone(df_nihchestxray14, 'Synthetic_Report', 'Image Index')
        # filter studies to use for current split
        filenames_nihchestxray14 = []
        for _, row in df_nihchestxray14.iterrows():
            path = getattr(row, 'Image Index')
            if path in path2sent_nihchestxray14:
                filenames_nihchestxray14.append(path)
        ########################################### Vindr-CXR #########################################################
        if not os.path.exists(Vindr_CXR_DATA_DIR):
            raise RuntimeError(f"{Vindr_CXR_DATA_DIR} does not exist!")
        df_VindrCXR = pd.read_csv(Vindr_CXR_TRAIN_CSV)  # 15000
        # all views in this dataset are frontal
        df_VindrCXR['image_id'] = df_VindrCXR['image_id'].apply(lambda path: os.path.join(Vindr_CXR_DATA_DIR, 'train_png', path + '.png'))
        path2sent_VindrCXR = create_path_2_sent_mapping_standalone(df_VindrCXR, 'Synthetic_Report', 'image_id')
        # filter studies to use for current split
        filenames_VindrCXR = []
        for row in df_VindrCXR.itertuples():
            path = getattr(row, 'image_id')
            if path in path2sent_VindrCXR:
                filenames_VindrCXR.append(path)
        ########################################### Vindr-PCXR ########################################################
        if not os.path.exists(Vindr_PCXR_DATA_DIR):
            raise RuntimeError(f"{Vindr_PCXR_DATA_DIR} does not exist!")
        df_VindrPCXR = pd.read_csv(Vindr_PCXR_TRAIN_CSV)  # 7728
        # all views in this dataset are frontal
        df_VindrPCXR['image_id'] = df_VindrPCXR['image_id'].apply(lambda path: os.path.join(Vindr_PCXR_DATA_DIR, 'train_png', path + '.png'))
        path2sent_VindrPCXR = create_path_2_sent_mapping_standalone(df_VindrPCXR, 'Synthetic_Report', 'image_id')
        # filter studies to use for current split
        filenames_VindrPCXR = []
        for row in df_VindrPCXR.itertuples():
            path = getattr(row, 'image_id')
            if path in path2sent_VindrPCXR:
                filenames_VindrPCXR.append(path)

        ########################################### combine all datasets ##############################################
        self.filenames = filenames_chexpert + \
                         filenames_mimic + \
                         filenames_padchest + \
                         filenames_openi + \
                         filenames_bimcv + \
                         filenames_casia + \
                         filenames_candid + \
                         filenames_ReXGradient + \
                         filenames_brax + \
                         filenames_ChestDR + \
                         filenames_nihchestxray14 + \
                         filenames_VindrCXR + \
                         filenames_VindrPCXR
        
        self.path2sent = {**path2sent_chexpert,
                          **path2sent_mimic,
                          **path2sent_padchest,
                          **path2sent_openi,
                          **path2sent_bimcv,
                          **path2sent_casia,
                          **path2sent_candid,
                          **path2sent_ReXGradient,
                          **path2sent_brax,
                          **path2sent_ChestDR,
                          **path2sent_nihchestxray14,
                          **path2sent_VindrCXR,
                          **path2sent_VindrPCXR
                          }

        self.all_filenames = self.filenames

    def set_subset(self, example_ids):
        """
        example_ids: indices relative to original full dataset
        """
        if isinstance(example_ids, set):
            example_ids = list(example_ids)
        self.filenames = [self.all_filenames[i] for i in example_ids]
        print(f"[Dataset] Using curated subset with {len(self.filenames)} samples.")

    def reset_subset(self):
        self.filenames = self.all_filenames
        print("[Dataset] Reset to full dataset.")

    def get_caption(self, path):
        series_sents = self.path2sent[path]
        return series_sents, len(series_sents)

    def get_imgs(self, img_path, transform=None, multiscale=False):
        image = Image.open(img_path).convert("RGB")
        img = transform(image)
        return img

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        key = self.filenames[index]
        caps, cap_len = self.get_caption(key)
        imgs = self.get_imgs(key, self.transform, multiscale=False)

        result = {"pixel_values": imgs, "image_ids": index, "texts": caps}

        return result

    def collate_fn(self, instances: List):
        collate = default_collate(instances)
        text_tokens = self.tokenizer(
            collate["texts"], padding="longest", truncation=True, return_tensors="pt", max_length=self.max_bert_length
        )
        collate["text_tokens"] = text_tokens

        return collate


def get_downstream_dataset(catalog, name, is_train, transform):
    entry = catalog[name]
    root = entry['path']
    if name == 'chexpert_test':
        dataset = CheXpertTestDataset(root_dir=entry['path'], gt=entry['gt_path'], transform=transform)   # 500 test samples
    elif name == 'chexpert_5x200':
        dataset = CheXpert5x200Dataset(root_dir=entry['path'], gt=entry['gt_path'], transform=transform)   # 1000 test samples
    elif name == 'mimic':
        dataset = MIMICCXRDataset(root_dir=entry['path'], gt=entry['gt_path'], transform=transform)   # 3082 test samples
    elif name == 'nihchestxray14':
        dataset = ChestXray14Dataset(root_dir=entry['path'], gt=entry['gt_path'], transform=transform)   # 25596 test samples
    elif name == 'pneumonia_Xray2017':
        dataset = Pneumonia_Xray2017(root_dir=entry['path'], transform=transform)   # 624 test samples  {'NORMAL': 0, 'PNEUMONIA': 1}
    elif name == 'simm_pneumothorax':
        dataset = SIIMPneumothoraxCXRDataset(root_dir=entry['path'], gt=entry['gt_path'], transform=transform)   # 1372 test samples
    elif name == 'tbx11k':
        dataset = TBCXRDataset(root_dir=entry['path'], gt=entry['gt_path'], transform=transform)   # 1800 test samples
    elif name == 'vindr_cxr':
        dataset = VinDrCXRDataset(root_dir=entry['path'], gt=entry['gt_path'], transform=transform)   # 3000 test samples
    elif name == 'vindr_pcxr':
        dataset = VinDrPCXRDataset(root_dir=entry['path'], gt=entry['gt_path'], transform=transform)   # 1397 test samples
    elif name == 'chexpert_5x200_retrieve':
        dataset = CheXpertRetrieveDataset(root_dir=entry['path'], gt=entry['gt_path'], transform=transform)   # 998 test samples
    elif name == 'mimic_retrieve':
        dataset = MIMICCXRRetrieveDataset(root_dir=entry['path'], gt=entry['gt_path'], transform=transform)   # 3082 test samples
    elif name == 'rexgraident_retrieve':
        dataset = ReXGraidentRetrieveDataset(root_dir=entry['path'], gt=entry['gt_path'], transform=transform)   # 8083 test samples
    else:
        raise Exception('Unknown dataset')                                                                                                                    

    return dataset

