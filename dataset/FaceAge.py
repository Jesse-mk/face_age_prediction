import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import cv2
from PIL import Image
import dlib
import pandas as pd
from transformers import ViTConfig, ViTModel, ViTFeatureExtractor
import torch.nn as nn
import os
from utils import generate_masks


class FaceAgeDataset(Dataset):
    def __init__(
        self,
        csv_path,
        data_dir="/home/jessekim/",
        transform=None,
        mask_info="BOX",
        data_source="wiki",
    ):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        shape_predictor_path = os.path.join(
            data_dir, "models", "shape_predictor_68_face_landmarks.dat"
        )
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)

        self.transform = transform
        self.mask_info = mask_info
        self.data_source = data_source
        print(self.data_source)

        ## VIT constants
        self.IMAGE_SIZE = 224  # 384 #224
        self.N_LABELS = 99  # 1 to 99 but subtract 1 so 0 to 98

        # transformations:
        #         self.preprocess_mask = transforms.Compose([
        #             transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
        #             transforms.ToTensor()])

        norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            'google/vit-base-patch16-224-in21k'
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        print(idx)
        # get specific example:
        sample = self.df.iloc[idx]


        if self.data_source == "wiki":
            img_path = os.path.join(
                self.data_dir, "data", "wiki_crop", sample["full_path"]
            )
        elif self.data_source == "lap":
            #should be train, valid or test
            train_type = self.csv_path.split("gt_")[1].split(".csv")[0]

            img_path = os.path.join(
                self.data_dir, "data", "appa-real-release", train_type, sample["file_name"] + "_face.jpg"
            )
        else:
            img_path = os.path.join(self.data_dir, "data", sample["full_path"])

        img = Image.open(img_path)
        tensor_img = transforms.ToTensor()(img)
        ### IMAGE CHANNEL/SIZE CONVERSIONS TO GET 3 CHANNEL IMAGES ##

        # convert to RGB
        if (len(tensor_img.shape) < 3) | (tensor_img.shape[0] != 3):  # if not RGB:
            img = img.convert("RGB")

        regions = False
        box = False

        if self.mask_info == "BOX":
            box = True
        elif self.mask_info == "5CHANNEL":
            regions = True

        # generate normalized and resized image:
        processed_img = self.feature_extractor(img)["pixel_values"][0]
        transformed_img = torch.tensor(processed_img)

        # generate masks
        mask = generate_masks(
            img_path, self.predictor, self.detector, regions=regions, box=box
        )
        transformed_mask = cv2.resize(mask, dsize=(self.IMAGE_SIZE, self.IMAGE_SIZE))
        transformed_mask = torch.tensor(transformed_mask)
        if len(transformed_mask.shape) < 3:
            transformed_mask = transformed_mask.unsqueeze(-1)

        # get label:
        try:
            label = (
                sample["age"] - 1
            )  # need to subtract one since need to fit in possible values!! #want to get label to be between 0 and 1 (normalize label)
        except:
            label = (
                sample["real_age"] - 1
            )

        return transformed_img, label, transformed_mask 
