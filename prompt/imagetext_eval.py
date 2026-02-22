import ast
from typing import Dict, List
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import default_collate
from torch.utils.data.dataset import Dataset
import cv2
# from cxrclip.data.data_utils import load_transform, transform_image
from .constants import CHEXPERT_CLASS_PROMPTS


# def load_transform(split: str = "train", transform_config: Dict = None):
#     assert split in {"train", "valid", "test", "aug"}
#
#     config = []
#     if transform_config:
#         if split in transform_config:
#             config = transform_config[split]
#     image_transforms = []
#
#     for name in config:
#         if hasattr(transforms, name):
#             tr_ = getattr(transforms, name)
#         else:
#             tr_ = getattr(albumentations, name)
#         tr = tr_(**config[name])
#         image_transforms.append(tr)
#
#     return image_transforms


# def transform_image(image_transforms, image: Union[Image.Image, np.ndarray], normalize="huggingface"):
#     for tr in image_transforms:
#         if isinstance(tr, albumentations.BasicTransform):
#             image = np.array(image) if not isinstance(image, np.ndarray) else image
#             image = tr(image=image)["image"]
#         else:
#             image = transforms.ToPILImage()(image) if not isinstance(image, Image.Image) else image
#             image = tr(image)
#
#     if normalize == "huggingface":
#         image = transforms.ToTensor()(image)
#         image = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)(image)
#
#     elif normalize == "imagenet":
#         image = transforms.ToTensor()(image)
#         image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
#
#     else:
#         raise KeyError(f"Not supported Normalize: {normalize}")
#
#     return image

def preprocess(img, desired_size=320):
    old_size = img.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = img.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_img = Image.new('L', (desired_size, desired_size))
    new_img.paste(img, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    return new_img

class ImageTextEvalDataset(Dataset):
    def __init__(
        self,
        name: str,
        data_path: str,
        split: str,
        data_frac: float = 1.0,
        tokenizer=None,
        text_max_length: int = 256,
        transform=None,
        normalize: str = "huggingface",
        **kwargs
    ):
        super().__init__()
        self.name = name
        self.split = split
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.data_frac = data_frac
        self.normalize = normalize

        if self.name == "chexpert5x200":
            self.label_list = list(CHEXPERT_CLASS_PROMPTS.keys())
        else:
            self.label_list = []

        self.idx2label = {idx: self.label_list[idx] for idx in range(len(self.label_list))}
        self.label2idx = {v: k for k, v in self.idx2label.items()}

        self.image_transforms = transform
        self.df = pd.read_csv(data_path)
        if data_frac < 1.0:
            self.df = self.df.sample(frac=self.data_frac, random_state=1, ignore_index=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.name == "chexpert5x200":
            image_path = self.df["Path"][index]
        else:
            image_path = self.df["image"][index]

        if image_path.startswith("["):
            image_path = ast.literal_eval(image_path)[0]  # not random sampling

        # read image using cv2
        img = cv2.imread(str('/mnt/c/chong/data/' + image_path))
        # convert to PIL Image object
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        # preprocess
        img = preprocess(img_pil, desired_size=320)

        img = np.expand_dims(np.array(img), axis=0)
        img = np.repeat(img, 3, axis=0)

        img = torch.from_numpy(img.astype(float)) # torch, (320, 320)

        image = self.image_transforms(img)

        if self.name == "chexpert5x200":
            text = self.df["Report Impression"][index]
        else:
            text = self.df["text"][index]

        sample = {"images": image, "text": text}

        if self.name in {"chexpert5x200"}:
            for label_candidate in self.label_list:
                if self.df[label_candidate][index] == 1.0:
                    label = label_candidate
            label_idx = self.label2idx[label]
            sample["label_names"] = label
            sample["label_indices"] = label_idx

        return sample

    def collate_fn(self, instances: List):
        collate = default_collate(instances)
        # text_tokens = self.tokenizer(
        #     collate["text"], padding="longest", truncation=True, return_tensors="pt", max_length=self.text_max_length
        # )
        # collate["text_tokens"] = text_tokens
        # texts = list([ins["text"] for ins in instances])
        # collate["texts"] = texts

        return collate
