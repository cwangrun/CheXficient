import os
from pathlib import Path


# #############################################
# CheXpert constants
# #############################################
CHEXPERT_DATA_DIR = Path('./data/CheXpert-v1.0')
CHEXPERT_TRAIN_CSV = CHEXPERT_DATA_DIR / "df_chexpert_plus_240401.csv"  # train split from train.csv
CHEXPERT_5x200 = CHEXPERT_DATA_DIR / "chexpert_5x200.csv"

CHEXPERT_PATH_COL = "Path"
CHEXPERT_SPLIT_COL = "Split"

CHEXPERT_TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# baseed on original chexpert paper
CHEXPERT_UNCERTAIN_MAPPINGS = {
    "Atelectasis": 1,
    "Cardiomegaly": 0,
    "Consolidation": 0,
    "Edema": 1,
    "Pleural Effusion": 1,
}

CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}


# #############################################
# MIMIC-CXR-JPG constants
# #############################################
MIMIC_CXR_DATA_DIR = Path("./data/mimic-cxr-jpg/2.0.0")
# Created csv
MIMIC_CXR_TRAIN_CSV = MIMIC_CXR_DATA_DIR / "train.csv"
MIMIC_CXR_VALID_CSV = MIMIC_CXR_DATA_DIR / "valid.csv"
MIMIC_CXR_TEST_CSV = MIMIC_CXR_DATA_DIR / "test.csv"
MIMIC_CXR_MASTER_CSV = MIMIC_CXR_DATA_DIR / "master.csv"
MIMIC_CXR_VIEW_COL = "ViewPosition"
MIMIC_CXR_PATH_COL = "Path"
MIMIC_CXR_SPLIT_COL = "split"

# #############################################
# PadChest constants
# #############################################
PadChest_CXR_DATA_DIR = Path("./data/padchest/")
PadChest_CXR_TRAIN_CSV = PadChest_CXR_DATA_DIR / "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"

# #############################################
# Open-I constants
# #############################################
Open_I_CXR_DATA_DIR = Path("./data/open-i/")
Open_I_CXR_TRAIN_CSV = Open_I_CXR_DATA_DIR / "indiana_reports.csv"

# #############################################
# BIMCV constants
# #############################################
BIMCV_CXR_DATA_DIR = Path("./data/bimcv-covid19/")
BIMCV_CXR_TRAIN_CSV = BIMCV_CXR_DATA_DIR / "final_annotations.csv"

# #############################################
# CANDID constants
# #############################################
CANDID_CXR_DATA_DIR = Path("./data/candid/")
CANDID_CXR_TRAIN_CSV = CANDID_CXR_DATA_DIR / "Pneumothorax_reports.csv"

# #############################################
# ReXGradient constants
# #############################################
ReXGradient_CXR_DATA_DIR = Path("./data/ReXGradient-160K/")
ReXGradient_CXR_TRAIN_CSV = ReXGradient_CXR_DATA_DIR / "train_metadata.csv"

# #############################################
# CASIA-CXR constants
# #############################################
CASIA_CXR_DATA_DIR = Path("./data/CASIA-CXR/")
CASIA_CXR_TRAIN_CSV = CASIA_CXR_DATA_DIR / "CASIA-CXR_Combined_Reports.csv"

# #############################################
# Brax constants
# #############################################
Brax_CXR_DATA_DIR = Path("./data/brax/")
Brax_CXR_TRAIN_CSV = Brax_CXR_DATA_DIR / "master_spreadsheet_update.csv"

# #############################################
# ChestDR constants
# #############################################
ChestDR_CXR_DATA_DIR = Path("./data/ChestDR/")
ChestDR_CXR_TRAIN_CSV = ChestDR_CXR_DATA_DIR / "chestdr_published_19d.csv"

# #############################################
# NIHChestXray14 constants
# #############################################
NIHChestXray14_CXR_DATA_DIR = Path("./data/chestxray14/")
NIHChestXray14_CXR_TRAIN_CSV = NIHChestXray14_CXR_DATA_DIR / "Data_Entry_2017.csv"

# #############################################
# Vindr-CXR constants
# #############################################
Vindr_CXR_DATA_DIR = Path("./data/vindr-cxr/")
Vindr_CXR_TRAIN_CSV = Vindr_CXR_DATA_DIR / "image_labels_train.csv"

# #############################################
# Vindr-PCXR constants
# #############################################
Vindr_PCXR_DATA_DIR = Path("./data/vindr-pcxr/")
Vindr_PCXR_TRAIN_CSV = Vindr_PCXR_DATA_DIR / "image_labels_train.csv"



