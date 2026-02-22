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


datapath = '/mnt/c/chong/data/padchest/'
csvpath = os.path.join(datapath, "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv")
df_csv = pd.read_csv(csvpath, low_memory=False)   # 160861

with open(os.path.join(datapath, "report.en.tok"), "r", encoding="utf-8") as f:
    english_report_list = f.readlines()

with open(os.path.join(datapath, "image.tok"), "r", encoding="utf-8") as f:
    image_list = f.readlines()

images, eng_report = [], []
for iii, paths in enumerate(image_list):
    paths = paths.strip().split(',')
    for path in paths:
        images.append(path)
        eng_report.append(english_report_list[iii])
image_series = pd.Series(images, name='ImageID')
eng_report_series = pd.Series(eng_report, name='Report_English')
df_english_report = pd.concat([image_series, eng_report_series], axis=1)

bad_path_without_english_report = []
for path in df_csv['ImageID']:
    if path not in images:
        bad_path_without_english_report.append(path)
bad_report = list(df_csv[df_csv['ImageID'].isin(bad_path_without_english_report)]["Report"])

master_df = pd.merge(df_csv, df_english_report, on="ImageID", how="left")
master_df.dropna(subset=["Report_English"], how="all", inplace=True)   # 160688

# for iii, path in enumerate(master_df[master_df["Projection"].isin(["COSTAL"])]["ImageID"]):
#     if os.path.isfile(datapath + 'images/' + path):
#         shutil.copy(datapath + 'images/' + path, './111/' + path)

# Standardize view names
# views_all = set(list(df_csv['Projection']))   # ['AP', 'AP_horizontal', 'COSTAL', 'EXCLUDE', 'L', 'PA', 'UNK']
view_dict = {}
for view in set(list(master_df['Projection'])):
    view_dict[view] = master_df[master_df["Projection"].isin([view])]
# master_df = master_df[master_df["Projection"].isin(['AP', 'AP_horizontal', 'PA', 'COSTAL'])]  # 111133
mask = master_df["ImageID"].apply(lambda path: os.path.isfile(os.path.join(datapath, 'images', path)))
final_df = master_df[mask]  # 111125
final_df.to_csv(csvpath.replace('.csv', '_chong.csv'), index=False)

print('Done!')


tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

reports = final_df['Report_English']

token_lengths = [len(tokenizer.encode(r, add_special_tokens=True)) for r in reports]

token_lengths = np.array(token_lengths)

print("📊 Bio_ClinicalBERT Token Length Stats:")
print(f"  5%：{np.percentile(token_lengths, 5):.2f}")
print(f"  avg     ：{np.mean(token_lengths):.2f}")
print(f"  mid   ：{np.median(token_lengths):.2f}")
print(f"  95%：{np.percentile(token_lengths, 95):.2f}")
print(f"  max   ：{np.max(token_lengths)}")

print('Done')







