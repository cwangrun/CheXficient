import os
from glob import glob
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from PIL import Image
import json
import shutil


# all_files = glob("/mnt/c/chong/data/cxr-data_padchest/padchest/images/*.*")
# size_all = []
# for file in all_files:
#     image = Image.open(file)
#     size = np.array(image.size)
#     size_all.append(size)
# size_all = np.array(size_all)
# print('Done', size_all.min(axis=0), size_all.max(axis=0))


datapath = '/mnt/c/chong/data/bimcv-covid19/'
csvpath = os.path.join(datapath, "final_annotations.csv")
df_csv = pd.read_csv(csvpath, low_memory=False)   # 73401

with open(os.path.join(datapath, "RRG/bimcv-covid19/test.report.en.tok"), "r", encoding="utf-8") as f:
    english_report_list = f.readlines()

with open(os.path.join(datapath, "RRG/bimcv-covid19/test.image.en.tok"), "r", encoding="utf-8") as f:
    image_list = f.readlines()

images, eng_report = [], []
for iii, paths in enumerate(image_list):
    paths = paths.strip().split(',')
    for path in paths:
        images.append(path)
        eng_report.append(english_report_list[iii])
image_series = pd.Series(images, name='file_path')
eng_report_series = pd.Series(eng_report, name='report_english')
df_english_report = pd.concat([image_series, eng_report_series], axis=1)

bad_path_without_english_report = []
for path in df_csv['file_path']:
    if path not in images:
        bad_path_without_english_report.append(path)
bad_report = list(df_csv[df_csv['file_path'].isin(bad_path_without_english_report)]["report"])
bad_report = [rrr for rrr in bad_report if rrr is not ' ']

master_df = pd.merge(df_csv, df_english_report, on="file_path", how="left")
master_df.dropna(subset=["report_english"], how="all", inplace=True)   # 65421

# for iii, path in enumerate(master_df[master_df["Projection"].isin(["COSTAL"])]["ImageID"]):
#     if os.path.isfile(datapath + 'images/' + path):
#         shutil.copy(datapath + 'images/' + path, './111/' + path)

# Standardize view names
# views_all = set(list(master_df['view']))   # ['ap', 'lateral', 'll', 'pa']
view_dict = {}
for view in set(list(master_df['view'])):
    view_dict[view] = master_df[master_df["view"].isin([view])]
# master_df = master_df[master_df["view"].isin(['ap', 'pa'])]  # 57316
mask = master_df["file_path"].apply(lambda path: os.path.isfile(os.path.join(datapath, 'images', path)))
final_df = master_df[mask]  # 65421
final_df.to_csv(csvpath.replace('.csv', '_chong.csv'), index=False)

print('Done!')


# 1. tokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

reports = final_df['report_english']

token_lengths = [len(tokenizer.encode(r, add_special_tokens=True)) for r in reports]


token_lengths = np.array(token_lengths)


print("📊 Bio_ClinicalBERT Token Length Stats:")
print(f"  5%：{np.percentile(token_lengths, 5):.2f}")
print(f"  avg     ：{np.mean(token_lengths):.2f}")
print(f"  mid   ：{np.median(token_lengths):.2f}")
print(f"  95%：{np.percentile(token_lengths, 95):.2f}")
print(f"  max   ：{np.max(token_lengths)}")

print('Done')









