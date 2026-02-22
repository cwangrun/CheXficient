import os
from glob import glob
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from PIL import Image
import json
import shutil
from pathlib import Path


# all_files = list(glob('/mnt/c/ReXGradient-160K/deid_png/*/*/*/*/*/*/*.*")'))
# size_all = []
# for file in all_files:
#     image = Image.open(file)
#     size = np.array(image.size)
#     size_all.append(size)
# size_all = np.array(size_all)
# print('Done', size_all.min(axis=0), size_all.max(axis=0))


datapath = "/mnt/c/chong/data/ReXGradient-160K/"

csvpath = os.path.join(datapath, "train_metadata.csv")
df_csv = pd.read_csv(csvpath, low_memory=False)   # 140000

# Open and read the JSON file
with open(os.path.join(datapath, 'train_metadata_view_position.json'), 'r') as f:
    meta_data = json.load(f)

ids, image_paths, image_views = [], [], []
for iii, (id, meta) in enumerate(meta_data.items()):
    paths = meta['ImagePath']
    views = meta['ImageViewPosition']
    for jjj, path in enumerate(paths):
        image_paths.append(path[3:])
        image_views.append(str(views[jjj]))
        ids.append(id)

id_series = pd.Series(ids, name='id')
path_series = pd.Series(image_paths, name='ImagePath')
view_series = pd.Series(image_views, name='ImageViewPosition')
df_meta = pd.concat([id_series, path_series, view_series], axis=1)  # 238968

master_df = pd.merge(df_meta, df_csv, on="id", how="left")
master_df.dropna(subset=["Findings", "Impression"], how="all", inplace=True)   # 238968
master_df[["Findings"]] = master_df[["Findings"]].fillna(" ")
master_df[["Impression"]] = master_df[["Impression"]].fillna(" ")

# Standardize view names
# views_all = set(list(df_csv['Projection']))   # ['ANTERO_POSTERIOR', 'AP', 'AP AXIAL', 'DECUBITUS', 'ERECT', 'KUB', 'LAO', 'LAT', 'LATERAL', 'LL', 'LPO', 'N/A', 'None', 'OBLIQUE', 'OTHER', 'PA', 'PICC LINE', 'POSTERO_ANTERIOR', 'RAO', 'RPO', 'SUPINE', 'UNKNOWN']
view_dict = {}
for view in set(list(master_df['ImageViewPosition'])):
    view_dict[view] = master_df[master_df["ImageViewPosition"].isin([view])]

# for try_path in view_dict['UNKNOWN']["ImagePath"]:    # 检查后，UNKNOWN (27835) 大部分（90%）也是 frontal
#     if os.path.isfile(os.path.join(datapath, try_path)):
#         shutil.copy(os.path.join(datapath, try_path), './try/' + os.path.basename(try_path))

# master_df = master_df[master_df["ImageViewPosition"].isin(['ANTERO_POSTERIOR', 'AP', 'AP AXIAL', 'DECUBITUS', 'ERECT', 'KUB', 'PA', 'PICC LINE', 'POSTERO_ANTERIOR', 'SUPINE', 'UNKNOWN'])]  # 140831
mask_exist = master_df["ImagePath"].apply(lambda path: os.path.isfile(os.path.join(datapath, path)))
final_df = master_df[mask_exist]  # 238968
final_df.to_csv(csvpath.replace('.csv', '_chong.csv'), index=False)

print('Done!')


tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

reports_findings = final_df['Findings'].tolist()
reports_impression = final_df['Impression'].tolist()
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

