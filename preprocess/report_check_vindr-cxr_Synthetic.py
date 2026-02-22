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


datapath = '/mnt/c/chong/data/vindr-cxr/'
csvpath = os.path.join(datapath, "image_labels_train.csv")
df_csv = pd.read_csv(csvpath, low_memory=False)   # 45000

label_columns = df_csv.columns.difference(['image_id', 'rad_id'])

merged_df = df_csv.groupby('image_id').agg({
    'rad_id': lambda x: list(x),
    **{col: 'sum' for col in label_columns}
}).reset_index()

merged_df[label_columns] = merged_df[label_columns].clip(upper=1)   # 15000


condition_label_map = {
    'No finding': 'No Finding',
    'Aortic enlargement': 'Aortic Enlargement',
    'Atelectasis': 'Atelectasis',
    'Calcification': 'Calcification',
    'Cardiomegaly': 'Cardiomegaly',
    'Clavicle fracture': 'Clavicle Fracture',
    'Consolidation': 'Consolidation',
    'Edema': 'Edema',
    'Emphysema': 'Emphysema',
    'Enlarged PA': 'Enlarged PA (Pulmonary Artery)',
    'ILD': 'ILD (Interstitial Lung Disease)',
    'Infiltration': 'Infiltration',
    'Lung Opacity': 'Lung Opacity',
    'Lung cavity': 'Lung Cavity',
    'Lung cyst': 'Lung Cyst',
    'Mediastinal shift': 'Mediastinal Shift',
    'Nodule/Mass': 'Nodule/mass',
    'Pleural effusion': 'Pleural Effusion',
    'Pleural thickening': 'Pleural Thickening',
    'Pneumothorax': 'Pneumothorax',
    'Pulmonary fibrosis': 'Pulmonary Fibrosis',
    'Rib fracture': 'Rib Fracture',
    'COPD': 'COPD (Chronic Obstructive Pulmonary Disease)',
    'Lung tumor': 'Lung Tumor',
    'Pneumonia': 'Pneumonia',
    'Tuberculosis': 'Tuberculosis',
    'Other lesion': 'Other Lesion',   ########   1154 positive
    'Other diseases': 'Other Diseases'  ########  4377 positive
}

conditions = list(condition_label_map.keys())


with open("report_templates.json", "r") as f:
    templates = json.load(f)


def generate_report(row):
    report = []

    if row.get("No finding") == 1.0 and all(pd.isna(row.get(c)) or row.get(c) == 0 for c in conditions if c != "No finding"):
        return random.choice(templates["no finding"])

    pos_index = []
    neg_index = []
    for ind, cond in enumerate(conditions[1:]):   # start from "Aortic Enlargement"
        val = row.get(cond)
        cond = condition_label_map[cond]
        if val == 1.0:
            sentence = random.choice(templates["positive"])
            sentence = sentence.format(condition=cond).lower().capitalize()
            report.append(sentence)
            pos_index.append(ind)
        elif val == 0.0:
            sentence = random.choice(templates["negative"])
            sentence = sentence.format(condition=cond).lower().capitalize()
            report.append(sentence)
            neg_index.append(ind)
        else:
            raise ValueError(f"Invalid value: {val}")
    use_index = sorted(pos_index + random.sample(neg_index, len(neg_index) - 7))  # only use a subset of negative sentences to limit 256 token length
    return " ".join([report[i] for i in use_index])


merged_df["Synthetic_Report"] = merged_df.apply(generate_report, axis=1)
merged_df.dropna(subset=["Synthetic_Report"], how="all", inplace=True)   # 15000

# all views in this dataset are frontal
mask = merged_df["image_id"].apply(lambda path: os.path.isfile(os.path.join(datapath, 'train_png', path + '.png')))
final_df = merged_df[mask]  # 15000
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







