import os
from glob import glob
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from PIL import Image
import json
import shutil


# all_files = glob("/mnt/c/chong/data/CheXpert-v1.0/train/*/*/*.*")
# size_all = []
# for file in all_files:
#     image = Image.open(file)
#     size = np.array(image.size)
#     size_all.append(size)
# size_all = np.array(size_all)
# print('Done', size_all.min(axis=0), size_all.max(axis=0))


datapath = '/mnt/c/chong/data/CheXpert-v1.0/'

csvpath = os.path.join(datapath, "df_chexpert_plus_240401.csv")
df_csv = pd.read_csv(csvpath, low_memory=False)   # 223462

exclude_csvpath = os.path.join(datapath, "chexpert_5x200.csv")
df_csv_exclude = pd.read_csv(exclude_csvpath, low_memory=False)
exclude_paths = df_csv_exclude['Path'].apply(lambda path: path.split('CheXpert-v1.0/')[1]).tolist()  # 1000
mask = df_csv["path_to_image"].apply(lambda path: path not in exclude_paths)
df_csv_new = df_csv[mask]  # 222464

mask_valid = df_csv_new["path_to_image"].apply(lambda path: path.startswith('train'))
df_csv_train = df_csv_new[mask_valid]  # 222230

df_csv_train.dropna(subset=["section_findings", "section_impression"], how="all", inplace=True)   # 222116
df_csv_train[["section_findings"]] = df_csv_train[["section_findings"]].fillna(" ")
df_csv_train[["section_impression"]] = df_csv_train[["section_impression"]].fillna(" ")

# for iii, path in enumerate(master_df[master_df["Projection"].isin(["COSTAL"])]["ImageID"]):
#     if os.path.isfile(datapath + 'images/' + path):
#         shutil.copy(datapath + 'images/' + path, './111/' + path)

# Standardize view names
# views_all = set(list(df_csv['Projection']))   # ['Frontal', 'Lateral']
view_dict = {}
for view in set(list(df_csv_train['frontal_lateral'])):
    view_dict[view] = df_csv_train[df_csv_train["frontal_lateral"].isin([view])]

# df_csv_train = df_csv_train[df_csv_train["frontal_lateral"].isin(['Frontal'])]  # 189774
mask_exist = df_csv_train["path_to_image"].apply(lambda path: os.path.isfile(os.path.join(datapath, path)))
final_df = df_csv_train[mask_exist]  # 222116
final_df.to_csv(csvpath.replace('.csv', '_chong.csv'), index=False)

print('Done!')



tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

reports_findings = final_df['section_findings'].tolist()
reports_impression = final_df['section_impression'].tolist()
reports = [str(find) + " " + str(impr) for find, impr in zip(reports_findings, reports_impression)]

token_lengths = [len(tokenizer.encode(r, add_special_tokens=True)) for r in reports]


token_lengths = np.array(token_lengths)


print("📊 Bio_ClinicalBERT Token Length Stats:")
print(f"  5%：{np.percentile(token_lengths, 5):.2f}")
print(f"  avg     ：{np.mean(token_lengths):.2f}")
print(f"  mid   ：{np.median(token_lengths):.2f}")
print(f"  95%：{np.percentile(token_lengths, 95):.2f}")
print(f"  max   ：{np.max(token_lengths)}")

print('Done')







