import os
from glob import glob
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from PIL import Image
import json, os
import shutil


# all_files = glob("/mnt/c/copy/data/open-i/NLMCXR_png/*.*")
# size_all = []
# for file in all_files:
#     image = Image.open(file)
#     size = np.array(image.size)
#     size_all.append(size)
# size_all = np.array(size_all)
# print('Done!', size_all.min(axis=0), size_all.max(axis=0))


all_files = glob('/mnt/c/chong/data/open-i/NLMCXR_dcm/*/*.*')  # 7470

datapath = '/mnt/c/chong/data/open-i/'

report_csvpath = os.path.join(datapath, "indiana_reports.csv")
df_csv = pd.read_csv(report_csvpath, low_memory=False)   # 3581

projection_csvpath = os.path.join(datapath, "indiana_projections.csv")
df_projection = pd.read_csv(projection_csvpath, low_memory=False)   # 7466

master_df = pd.merge(df_projection, df_csv, on="uid", how="left")
master_df.dropna(subset=["findings", 'impression'], how="all", inplace=True)   # 7426
master_df[["findings"]] = master_df[["findings"]].fillna(" ")
master_df[["impression"]] = master_df[["impression"]].fillna(" ")

# for iii, path in enumerate(master_df[master_df["Projection"].isin(["COSTAL"])]["ImageID"]):
#     if os.path.isfile(datapath + 'images/' + path):
#         shutil.copy(datapath + 'images/' + path, './111/' + path)

# Standardize view names
# views_all = set(list(df_csv['Projection']))   # ['Frontal', 'Lateral']
view_dict = {}
for view in set(list(master_df['projection'])):
    view_dict[view] = master_df[master_df["projection"].isin([view])]
# master_df = master_df[master_df["projection"].isin(['Frontal'])]  # 3794
master_df["filename"] = master_df["filename"].apply(lambda path: 'CXR' + path.replace('.dcm.', '.'))
mask = master_df["filename"].apply(lambda path: os.path.isfile(os.path.join(datapath, 'NLMCXR_png', path)))
final_df = master_df[mask]  # 7424
final_df.to_csv(report_csvpath.replace('.csv', '_chong.csv'), index=False)

print('Done!')




tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

reports_findings = final_df['findings'].tolist()
reports_impression = final_df['impression'].tolist()
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







