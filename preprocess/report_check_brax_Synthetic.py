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


datapath = '/mnt/c/chong/data/brax/'
csvpath = os.path.join(datapath, "master_spreadsheet_update.csv")
df_csv = pd.read_csv(csvpath, low_memory=False)   # 40967

# 标签字段
# conditions = \
# [
#     'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation',
#     'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
# ]
conditions = tuple(df_csv.columns)[8:-4]


with open("report_templates.json", "r") as f:
    templates = json.load(f)


# 合成报告函数
def generate_report(row):
    report = []

    # 如果“No Finding”是1且其他都没标注，则直接写入
    if row.get("No Finding") == 1.0 and all(pd.isna(row.get(c)) or row.get(c) == 0 for c in conditions if c != "No Finding"):
        return random.choice(templates["no finding"])

    for cond in conditions[1:]:   # start from "Enlarged Cardiomediastinum"
        val = row.get(cond)
        if pd.isna(val):
            continue
        if val == 1.0:
            if cond == "Pleural Other":
                sentence = random.choice(templates["pleural others"])
            else:
                sentence = random.choice(templates["positive"])
                sentence = sentence.format(condition=cond).lower().capitalize()
            report.append(sentence)
        elif val == 0.0:
            if cond == "Pleural Other":
                sentence = random.choice(templates["no pleural others"])
            elif cond == "Support Devices":
                sentence = random.choice(templates["no support devices"])
            else:
                sentence = random.choice(templates["negative"])
                sentence = sentence.format(condition=cond).lower().capitalize()
            report.append(sentence)
        elif val == -1.0:
            if cond == "Pleural Other":
                sentence = random.choice(templates["uncertain pleural others"])
            else:
                sentence = random.choice(templates["uncertain"])
                sentence = sentence.format(condition=cond).lower().capitalize()
            report.append(sentence)
        else:
            raise ValueError(f"Invalid value: {val}")

    return " ".join(report)


# 应用生成器
df_csv["Synthetic_Report"] = df_csv.apply(generate_report, axis=1)
df_csv.dropna(subset=["Synthetic_Report"], how="all", inplace=True)   # 40967

# views_all = set(list(df_csv['view']))   # ['PA', 'RL', 'AP LLD', 'AP', 'L', nan, 'RLO', 'LT-DECUB']
view_dict = {}
for view in set((list(df_csv['ViewPosition']))):
    view_dict[view] = df_csv[df_csv["ViewPosition"].isin([view])]

# df_csv = df_csv[df_csv["ViewPosition"].isin(['PA', 'AP LLD', 'AP'])]  # 19310
mask = df_csv["PngPath"].apply(lambda path: os.path.isfile(os.path.join(datapath, path)))
final_df = df_csv[mask]  # 19310
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








