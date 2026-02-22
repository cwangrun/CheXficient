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


datapath = '/mnt/c/chong/data/candid/'
csvpath = os.path.join(datapath, "Pneumothorax_reports.csv")
df_csv = pd.read_csv(csvpath, low_memory=False)   # 19640

with open(os.path.join(datapath, "RRG/candid-ptx/impression.tok"), "r", encoding="utf-8") as f:
    english_report_list = f.readlines()

with open(os.path.join(datapath, "RRG/candid-ptx/image.tok"), "r", encoding="utf-8") as f:
    image_list = f.readlines()

images, SOPInstanceUID, impression_report = [], [], []
for iii, paths in enumerate(image_list):
    paths = paths.strip().split(',')
    for path in paths:
        images.append(path)
        SOPInstanceUID.append(os.path.splitext(os.path.basename(path))[0])
        impression_report.append(english_report_list[iii])

image_series = pd.Series(images, name='file_path')
instanceUID_series = pd.Series(SOPInstanceUID, name='SOPInstanceUID')
eng_report_series = pd.Series(impression_report, name='report_impression')
df_english_report = pd.concat([image_series, instanceUID_series, eng_report_series], axis=1)

bad_path_without_english_report = []
for path in df_csv['SOPInstanceUID']:
    if path not in SOPInstanceUID:
        bad_path_without_english_report.append(path)
bad_report = list(df_csv[df_csv['SOPInstanceUID'].isin(bad_path_without_english_report)]["Report"])

master_df = pd.merge(df_csv, df_english_report, on="SOPInstanceUID", how="left")
master_df.dropna(subset=["report_impression"], how="all", inplace=True)   # 19609

# all view in this dataset are frontal (ap, pa)
mask = master_df["file_path"].apply(lambda path: os.path.isfile(os.path.join(datapath, path)))
final_df = master_df[mask]  # 19609
final_df.to_csv(csvpath.replace('.csv', '_chong.csv'), index=False)

print('Done!')


tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

reports = final_df['report_impression']
token_lengths = [len(tokenizer.encode(r, add_special_tokens=True)) for r in reports]

token_lengths = np.array(token_lengths)


print("📊 Bio_ClinicalBERT Token Length Stats:")
print(f"  5%：{np.percentile(token_lengths, 5):.2f}")
print(f"  avg     ：{np.mean(token_lengths):.2f}")
print(f"  mid   ：{np.median(token_lengths):.2f}")
print(f"  95%：{np.percentile(token_lengths, 95):.2f}")
print(f"  max   ：{np.max(token_lengths)}")

print('Done')









