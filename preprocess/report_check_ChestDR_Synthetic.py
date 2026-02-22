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


datapath = '/mnt/c/chong/data/ChestDR/'
csvpath = os.path.join(datapath, "chestdr_published_19d.csv")
df_csv = pd.read_csv(csvpath, low_memory=False)   # 4848


condition_label_map = {
    'pleural_effusion': 'Pleural Effusion',
    'nodule': 'Nodule',
    'pneumonia': 'Pneumonia',
    'cardiomegaly': 'Cardiomegaly',
    'hilar_enlargement': 'Hilar Enlargement',
    'fracture_old': 'Old Fracture',
    'fibrosis': 'Fibrosis',
    'aortic_calcification': 'Aortic Calcification',
    'tortuous_aorta': 'Tortuous Aorta',
    'thickened_pleura': 'Thickened Pleura',
    'TB': 'TB (Tuberculosis)',
    'pneumothorax': 'Pneumothorax',
    'emphysema': 'Emphysema',
    'atelectasis': 'Atelectasis',
    'calcification': 'Calcification',
    'pulmonary_edema': 'Pulmonary Edema',
    'increased_lung_markings': 'Increased Lung Markings',
    'elevated_diaphragm': 'Elevated Diaphragm',
    'consolidation': 'Consolidation'
}

conditions = list(condition_label_map.keys())


with open("report_templates.json", "r") as f:
    templates = json.load(f)


def generate_report(row):
    report = []

    for cond in conditions:
        val = row.get(cond)
        cond = condition_label_map[cond]
        if pd.isna(val):
            continue
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
df_csv.dropna(subset=["Synthetic_Report"], how="all", inplace=True)   # 4848

# all views in this dataset are frontal
mask = df_csv["img_id"].apply(lambda path: os.path.isfile(os.path.join(datapath, 'images', path)))
final_df = df_csv[mask]  # 4848
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







