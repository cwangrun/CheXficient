import pandas as pd
from fun_utils import extract_mimic_text
import numpy as np


np.random.seed(42)


extract_text = False
# extract_text = True

MIMIC_CXR_META_CSV = '/mnt/c/chong/data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv'
MIMIC_CXR_TEXT_CSV = '/mnt/c/chong/data/mimic-cxr-jpg/2.0.0/yulequan/mimic_cxr_sectioned.csv'
MIMIC_CXR_SPLIT_CSV = '/mnt/c/chong/data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv'
MIMIC_CXR_CHEXPERT_CSV = '/mnt/c/chong/data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv'
MIMIC_CXR_DATA_DIR = "/mnt/c/chong/data/mimic-cxr-jpg/2.0.0"


MIMIC_CXR_TRAIN_CSV = '/mnt/c/chong/data/mimic-cxr-jpg/2.0.0/yulequan/train.csv'
MIMIC_CXR_VALID_CSV = '/mnt/c/chong/data/mimic-cxr-jpg/2.0.0/yulequan/valid.csv'
MIMIC_CXR_TEST_CSV = '/mnt/c/chong/data/mimic-cxr-jpg/2.0.0/yulequan/test.csv'
MIMIC_CXR_MASTER_CSV = '/mnt/c/chong/data/mimic-cxr-jpg/2.0.0/yulequan/master.csv'

def main():

    if extract_text:
        extract_mimic_text()

    metadata_df = pd.read_csv(MIMIC_CXR_META_CSV)
    metadata_df = metadata_df[["dicom_id", "subject_id", "study_id", "ViewPosition"]].astype(str)
    metadata_df["study_id"] = metadata_df["study_id"].apply(lambda x: "s"+x)   # 377110
    # Only keep frontal images
    # views_all = set(list(metadata_df['ViewPosition']))   # ['AP', 'AP AXIAL', 'AP LLD', 'AP RLD', 'LAO', 'LATERAL', 'LL', 'LPO', 'PA', 'PA LLD', 'PA RLD', 'RAO', 'SWIMMERS','XTABLE LATERAL', nan]
    # metadata_df = metadata_df[metadata_df["ViewPosition"].isin(["AP", 'PA'])]   # 243334 yulequan
    # metadata_df = metadata_df[metadata_df["ViewPosition"].isin(["AP", 'AP AXIAL', 'AP LLD', 'AP RLD', 'PA', 'PA LLD', 'PA RLD'])]     # 243345    ############ by chong

    view_dict = {}
    for view in set(list(metadata_df['ViewPosition'])):
        view_dict[view] = metadata_df[metadata_df["ViewPosition"].isin([view])]

    text_df = pd.read_csv(MIMIC_CXR_TEXT_CSV)
    text_df.dropna(subset=["impression", "findings"], how="all", inplace=True)
    text_df = text_df[["study", "impression", "findings"]]
    text_df.rename(columns={"study": "study_id"}, inplace=True)

    split_df = pd.read_csv(MIMIC_CXR_SPLIT_CSV)
    split_df = split_df.astype(str)
    split_df["study_id"] = split_df["study_id"].apply(lambda x: "s"+x)
    # TODO: merge validate and test into test.
    # split_df["split"] = split_df["split"].apply(lambda x: "valid" if x == "validate" or x == "test" else x)
    split_df["split"] = split_df["split"].apply(lambda x: "valid" if x == "validate" else x)

    chexpert_df = pd.read_csv(MIMIC_CXR_CHEXPERT_CSV)
    chexpert_df[["subject_id", "study_id"]] = chexpert_df[["subject_id", "study_id"]].astype(str)
    chexpert_df["study_id"] = chexpert_df["study_id"].apply(lambda x: "s"+x)

    master_df = pd.merge(metadata_df, text_df, on="study_id", how="left")
    master_df = pd.merge(master_df, split_df, on=["dicom_id", "subject_id", "study_id"], how="inner")
    master_df.dropna(subset=["impression", "findings"], how="all", inplace=True)
    
    n = len(master_df)
    master_data = master_df.values

    root_dir = str(MIMIC_CXR_DATA_DIR).split("/")[-1] + "/files"
    path_list = []
    for i in range(n):
        row = master_data[i]
        file_path = "%s/p%s/p%s/%s/%s.jpg" % (root_dir, str(row[1])[:2], str(row[1]), str(row[2]), str(row[0]))
        path_list.append(file_path)

    master_df.insert(loc=0, column="Path", value=path_list)

    # Create labeled data df
    labeled_data_df = pd.merge(master_df, chexpert_df, on=["subject_id", "study_id"], how="inner")
    labeled_data_df.drop(["dicom_id", "subject_id", "study_id", "impression", "findings"], axis=1, inplace=True)

    train_df = labeled_data_df.loc[labeled_data_df["split"] == "train"]
    train_df.to_csv(MIMIC_CXR_TRAIN_CSV, index=False)
    valid_df = labeled_data_df.loc[labeled_data_df["split"] == "valid"]
    valid_df.to_csv(MIMIC_CXR_VALID_CSV, index=False)
    test_df = labeled_data_df.loc[labeled_data_df["split"] == "test"]
    test_df.to_csv(MIMIC_CXR_TEST_CSV, index=False)

    # master_df.drop(["dicom_id", "subject_id", "study_id"], axis=1, inplace=True)

    # Fill nan in text
    master_df[["impression"]] = master_df[["impression"]].fillna(" ")
    master_df[["findings"]] = master_df[["findings"]].fillna(" ")
    master_df.to_csv(MIMIC_CXR_MASTER_CSV, index=False)


if __name__ == "__main__":
    main()