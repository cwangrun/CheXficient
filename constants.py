import os
from pathlib import Path


# DATA_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
# DATA_BASE_DIR = '/mnt/c/chong/data/'
# DATA_BASE_DIR = Path(DATA_BASE_DIR)

# #############################################
# CheXpert constants
# #############################################
CHEXPERT_DATA_DIR = Path('/mnt/c/chong/data/CheXpert-v1.0')
CHEXPERT_ORIGINAL_TRAIN_CSV = CHEXPERT_DATA_DIR / "train.csv"
CHEXPERT_TRAIN_CSV = CHEXPERT_DATA_DIR / "df_chexpert_plus_240401_chong.csv"  # train split from train.csv
CHEXPERT_VALID_CSV = CHEXPERT_DATA_DIR / "valid_split.csv"  # valid split from train.csv
CHEXPERT_TEST_CSV = (CHEXPERT_DATA_DIR / "valid.csv")  # using validation set as test set (test set label hidden)
CHEXPERT_MASTER_CSV = (CHEXPERT_DATA_DIR / "master_updated.csv")  # contains patient information, not PHI conplient
CHEXPERT_TRAIN_DIR = CHEXPERT_DATA_DIR / "train"
CHEXPERT_TEST_DIR = CHEXPERT_DATA_DIR / "valid"
CHEXPERT_5x200 = CHEXPERT_DATA_DIR / "chexpert_5x200.csv"
CHEXPERT_8x200_QUERY = CHEXPERT_DATA_DIR / "chexpert_8x200_query.csv"
CHEXPERT_8x200_CANDIDATES = CHEXPERT_DATA_DIR / "chexpert_8x200_candidates.csv"

CHEXPERT_VALID_NUM = 5000
CHEXPERT_VIEW_COL = "Frontal/Lateral"
CHEXPERT_PATH_COL = "Path"
CHEXPERT_SPLIT_COL = "Split"
CHEXPERT_REPORT_COL = "Report Impression"

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
CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# baseed on original chexpert paper
CHEXPERT_UNCERTAIN_MAPPINGS = {
    "Atelectasis": 1,
    "Cardiomegaly": 0,
    "Consolidation": 0,
    "Edema": 1,
    "Pleural Effusion": 1,
}

# CHEXPERT_CLASS_PROMPTS = {
#     "Atelectasis": "Platelike opacity likely represents atelectasis.",
#     "Cardiomegaly": "The cardiac silhouette is enlarged.",
#     "Edema": "The presence of hazy opacity suggests interstitial pulmonary edema.",
#     "Fracture": "A cortical step off indicates the presence of a fracture.",
#     "Pleural Effusion": "The pleural space is partially filled with fluid",
#     "Pneumonia": "A pulmonary opacity with ill defined borders likely represents pneumonia.",
#     "Pneumothorax": "A medial pneumothorax is present adjacent to the heart.",
#     "No Finding": "No clinically significant radiographic abnormalities."
# }

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
MIMIC_CXR_DATA_DIR = Path("/mnt/c/chong/data/mimic-cxr-jpg/2.0.0")
# MIMIC_CXR_TRAIN_TXT = MIMIC_CXR_DATA_DIR / "train.txt"
# MIMIC_CXR_VALID_TXT = MIMIC_CXR_DATA_DIR / "test.txt"
MIMIC_CXR_CHEXPERT_CSV = MIMIC_CXR_DATA_DIR / "mimic-cxr-2.0.0-chexpert.csv"
MIMIC_CXR_META_CSV = MIMIC_CXR_DATA_DIR / "mimic-cxr-2.0.0-metadata.csv"
MIMIC_CXR_TEXT_CSV = MIMIC_CXR_DATA_DIR / "yulequan/mimic_cxr_sectioned.csv"
MIMIC_CXR_SPLIT_CSV = MIMIC_CXR_DATA_DIR / "mimic-cxr-2.0.0-split.csv"
# Created csv
MIMIC_CXR_TRAIN_CSV = MIMIC_CXR_DATA_DIR / "train.csv"
MIMIC_CXR_VALID_CSV = MIMIC_CXR_DATA_DIR / "test.csv"
MIMIC_CXR_TEST_CSV = MIMIC_CXR_DATA_DIR / "test.csv"
MIMIC_CXR_MASTER_CSV = MIMIC_CXR_DATA_DIR / "yulequan/master.csv"
MIMIC_CXR_VIEW_COL = "ViewPosition"
MIMIC_CXR_PATH_COL = "Path"
MIMIC_CXR_SPLIT_COL = "split"

# #############################################
# PadChest constants
# #############################################
PadChest_CXR_DATA_DIR = Path("/mnt/c/chong/data/padchest/")
PadChest_CXR_TRAIN_CSV = PadChest_CXR_DATA_DIR / "PADCHEST_chest_x_ray_images_labels_160K_01.02.19_chong.csv"

# #############################################
# Open-I constants
# #############################################
Open_I_CXR_DATA_DIR = Path("/mnt/c/chong/data/open-i/")
Open_I_CXR_TRAIN_CSV = Open_I_CXR_DATA_DIR / "indiana_reports_chong.csv"

# #############################################
# BIMCV constants
# #############################################
BIMCV_CXR_DATA_DIR = Path("/mnt/c/chong/data/bimcv-covid19/")
BIMCV_CXR_TRAIN_CSV = BIMCV_CXR_DATA_DIR / "final_annotations_chong.csv"

# #############################################
# CANDID constants
# #############################################
CANDID_CXR_DATA_DIR = Path("/mnt/c/chong/data/candid/")
CANDID_CXR_TRAIN_CSV = CANDID_CXR_DATA_DIR / "Pneumothorax_reports_chong.csv"

# #############################################
# ReXGradient constants
# #############################################
ReXGradient_CXR_DATA_DIR = Path("/mnt/c/chong/data/ReXGradient-160K/")
ReXGradient_CXR_TRAIN_CSV = ReXGradient_CXR_DATA_DIR / "train_metadata_chong.csv"

# #############################################
# CASIA-CXR constants
# #############################################
CASIA_CXR_DATA_DIR = Path("/mnt/c/chong/data/CASIA-CXR/")
CASIA_CXR_TRAIN_CSV = CASIA_CXR_DATA_DIR / "CASIA-CXR_Combined_Reports_chong.csv"

# #############################################
# Brax constants
# #############################################
Brax_CXR_DATA_DIR = Path("/mnt/c/chong/data/brax/")
Brax_CXR_TRAIN_CSV = Brax_CXR_DATA_DIR / "master_spreadsheet_update_chong.csv"

# #############################################
# ChestDR constants
# #############################################
ChestDR_CXR_DATA_DIR = Path("/mnt/c/chong/data/ChestDR/")
ChestDR_CXR_TRAIN_CSV = ChestDR_CXR_DATA_DIR / "chestdr_published_19d_chong.csv"

# #############################################
# NIHChestXray14 constants
# #############################################
NIHChestXray14_CXR_DATA_DIR = Path("/mnt/c/chong/data/chestxray14/")
NIHChestXray14_CXR_TRAIN_CSV = NIHChestXray14_CXR_DATA_DIR / "Data_Entry_2017_chong.csv"

# #############################################
# Vindr-CXR constants
# #############################################
Vindr_CXR_DATA_DIR = Path("/mnt/c/chong/data/vindr-cxr/")
Vindr_CXR_TRAIN_CSV = Vindr_CXR_DATA_DIR / "image_labels_train_chong.csv"

# #############################################
# Vindr-PCXR constants
# #############################################
Vindr_PCXR_DATA_DIR = Path("/mnt/c/chong/data/vindr-pcxr/")
Vindr_PCXR_TRAIN_CSV = Vindr_PCXR_DATA_DIR / "image_labels_train_chong.csv"







# #############################################
# RSNA constants
# #############################################
RSNA_DATA_DIR = Path("/mnt/c/chong/data/RSNA_Pneumonia")
RSNA_ORIGINAL_TRAIN_CSV = RSNA_DATA_DIR / "stage_2_train_labels.csv"
RSNA_CLASSINFO_CSV = RSNA_DATA_DIR / "stage_2_detailed_class_info.csv"
RSNA_TRAIN_CSV = RSNA_DATA_DIR / "train.csv"
RSNA_VALID_CSV = RSNA_DATA_DIR / "val.csv"
RSNA_TEST_CSV = RSNA_DATA_DIR / "test.csv"
RSNA_DETECTION_TRAIN_PKL = RSNA_DATA_DIR / "train.pkl"
RSNA_DETECTION_VALID_PKL = RSNA_DATA_DIR / "val.pkl"
RSNA_DETECTION_TEST_PKL = RSNA_DATA_DIR / "test.pkl"

RSNA_IMG_DIR = RSNA_DATA_DIR / "stage_2_train_images"
RSNA_TRAIN_PCT = 0.7


# #############################################
# SIIM constants
# #############################################
PNEUMOTHORAX_DATA_DIR = Path("/mnt/c/chong/data/SIIM_Pneumothorax")
PNEUMOTHORAX_ORIGINAL_TRAIN_CSV = PNEUMOTHORAX_DATA_DIR / "train-rle.csv"
PNEUMOTHORAX_TRAIN_CSV = PNEUMOTHORAX_DATA_DIR / "train.csv"
PNEUMOTHORAX_VALID_CSV = PNEUMOTHORAX_DATA_DIR / "valid.csv"
PNEUMOTHORAX_TEST_CSV = PNEUMOTHORAX_DATA_DIR / "test.csv"
PNEUMOTHORAX_IMG_DIR = PNEUMOTHORAX_DATA_DIR / "dicom-images-train"
PNEUMOTHORAX_IMG_SIZE = 1024
PNEUMOTHORAX_TRAIN_PCT = 0.7


# #############################################
# tuberculosis constants
# #############################################
COVIDX_DATA_DIR = Path("/mnt/c/chong/data/COVIDx")
# COVIDX_ORIGINAL_TRAIN_TXT = COVIDX_DATA_DIR / "train.txt"
COVIDX_ORIGINAL_TRAIN_TXT = COVIDX_DATA_DIR / "train_COVIDx9A.txt"
# COVIDX_ORIGINAL_TEST_TXT = COVIDX_DATA_DIR / "test.txt"
COVIDX_ORIGINAL_TEST_TXT = COVIDX_DATA_DIR / "test_COVIDx9A.txt"
COVIDX_TRAIN_CSV = COVIDX_DATA_DIR / "train.csv"
COVIDX_VALID_CSV = COVIDX_DATA_DIR / "valid.csv"
COVIDX_TEST_CSV = COVIDX_DATA_DIR / "test.csv"

# #############################################
# COVIDx constants
# #############################################
TUBERCULOSIS_DATA_DIR = Path("/mnt/c/chong/data/tuberculosis")
TUBERCULOSIS_ORIGINAL_TRAIN_CSV = TUBERCULOSIS_DATA_DIR / "shenzhen_metadata.csv"
TUBERCULOSIS_TRAIN_CSV = TUBERCULOSIS_DATA_DIR / "train.csv"
TUBERCULOSIS_VALID_CSV = TUBERCULOSIS_DATA_DIR / "valid.csv"
TUBERCULOSIS_TEST_CSV = TUBERCULOSIS_DATA_DIR / "test.csv"

# #############################################
# Vinbigdata constants
# #############################################
VIN_DATA_DIR = Path("/mnt/c/chong/data/vinbigdata")
VIN_ORIGINAL_TRAIN_TXT = VIN_DATA_DIR / "train.csv"
VIN_TRAIN_CSV = VIN_DATA_DIR / "train_df.csv"
VIN_VALID_CSV = VIN_DATA_DIR / "valid_df.csv"
VIN_TEST_CSV = VIN_DATA_DIR / "test_df.csv"


# #############################################
# Object CXR constants
# #############################################
OBJ_DATA_DIR =  Path("/mnt/c/chong/data/object-CXR")
OBJ_ORIGINAL_TRAIN_CSV = OBJ_DATA_DIR / "train.csv"
OBJ_ORIGINAL_DEV_CSV = OBJ_DATA_DIR / "dev.csv"
OBJ_TRAIN_PKL = OBJ_DATA_DIR / "train.pkl"
OBJ_VALID_PKL = OBJ_DATA_DIR / "valid.pkl"
OBJ_TEST_PKL = OBJ_DATA_DIR / "test.pkl"
OBJ_TRAIN_IMG_PATH = OBJ_DATA_DIR / "train"
OBJ_VALID_IMG_PATH = OBJ_DATA_DIR / "train"
OBJ_TEST_IMG_PATH = OBJ_DATA_DIR / "dev"