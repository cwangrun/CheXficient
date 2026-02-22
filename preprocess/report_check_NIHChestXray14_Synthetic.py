import os
from glob import glob
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from PIL import Image
import json
import shutil
import math
import random
import json
from sklearn.preprocessing import MultiLabelBinarizer



seed = 1
random.seed(seed)  # python random seed
np.random.seed(seed)  # numpy random seed


# all_files = glob("/mnt/c/chong/data/cxr-data_padchest/padchest/images/*.*")
# size_all = []
# for file in all_files:
#     image = Image.open(file)
#     size = np.array(image.size)
#     size_all.append(size)
# size_all = np.array(size_all)
# print('Done', size_all.min(axis=0), size_all.max(axis=0))


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

conditions = list(Labels.keys())

datapath = '/mnt/c/chong/data/chestxray14/'
csvpath = os.path.join(datapath, "Data_Entry_2017.csv")
df_csv = pd.read_csv(csvpath, low_memory=False)            # 112120

grond_truth = np.zeros((len(df_csv), len(Labels)))
for idx, findings in enumerate(df_csv['Finding Labels']):
    target = findings.split("|")
    binary_result = mlb.fit_transform([[Labels[i] for i in target]]).squeeze()
    grond_truth[idx] = binary_result

for label in reversed(Labels.keys()):
    df_csv.insert(loc=2, column=label, value=grond_truth[:, Labels[label]])


with open("report_templates.json", "r") as f:
    templates = json.load(f)


def generate_report(row):
    report = []

    if row.get("No Finding") == 1.0 and all(pd.isna(row.get(c)) or row.get(c) == 0 for c in conditions if c != "No Finding"):
        return random.choice(templates["no finding"])

    for cond in conditions[1:]:   # start from "Atelectasis"
        val = row.get(cond)
        cond = "Pleural Thickening" if cond == "Pleural_Thickening" else cond
        # if pd.isna(val):
        #     continue
        if val == 1.0:
            sentence = random.choice(templates["positive"])
            sentence = sentence.format(condition=cond).lower().capitalize()
            report.append(sentence)
        elif val == 0.0:
            sentence = random.choice(templates["negative"])
            sentence = sentence.format(condition=cond).lower().capitalize()
            report.append(sentence)
        else:
            raise ValueError(f"Invalid value: {val}")

    return " ".join(report)


df_csv["Synthetic_Report"] = df_csv.apply(generate_report, axis=1)
df_csv.dropna(subset=["Synthetic_Report"], how="all", inplace=True)   # 112120


label_file = 'train_val_list.txt'
img_list = os.path.join(datapath, label_file)
with open(img_list) as f:
    all_names = f.read().splitlines()
mask_train = df_csv["Image Index"].apply(lambda path: path in all_names)
df_csv = df_csv[mask_train]    # 86524


# all views in this dataset are frontal
mask = df_csv["Image Index"].apply(lambda path: os.path.isfile(os.path.join(datapath, 'data', path)))
final_df = df_csv[mask]  # 86524
final_df.to_csv(csvpath.replace('.csv', '_chong.csv'), index=False)

print('Done!')


tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

reports = final_df['Synthetic_Report']
token_lengths = [len(tokenizer.encode(r, add_special_tokens=True)) for r in reports]

token_lengths = np.array(token_lengths)


print("📊 Bio_ClinicalBERT Token Length Stats:")
print(f"  5%：{np.percentile(token_lengths, 5):.2f}")
print(f"  avg     ：{np.mean(token_lengths):.2f}")
print(f"  mid   ：{np.median(token_lengths):.2f}")
print(f"  95%：{np.percentile(token_lengths, 95):.2f}")
print(f"  max   ：{np.max(token_lengths)}")

print('Done')







