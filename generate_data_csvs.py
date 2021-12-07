import os
import sys
import pandas as pd
import numpy as np
import scipy.io as so
from datetime import datetime
import datetime as dt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch

# pip install opencv-python
import cv2
from PIL import Image
import imutils
from torch.optim import Adam
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
import torch.nn as nn
from transformers import Trainer
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import get_linear_schedule_with_warmup
from utils import generate_masks
import dlib
import warnings
import argparse

warnings.filterwarnings("ignore", category=UserWarning)


def generate_df_from_mat(
    data_dir, csv_dir, logs_dir, csv_path, feature_extractor, dataset_type="wiki"
):
    metadata_path = os.path.join(data_dir, dataset_type + ".mat")

    meta = so.loadmat(metadata_path)
    data = meta[dataset_type]

    # parse through matlab file
    dob = data[0][0][0].flatten()
    photo_taken = data[0][0][1].flatten()
    full_path = np.array([i[0] for i in data[0][0][2].flatten()])
    gender = data[0][0][3].flatten()
    name = [i[0] if len(i) > 0 else "" for i in data[0][0][4].flatten()]
    face_location = [i[0] for i in data[0][0][5].flatten()]
    face_score = data[0][0][6].flatten()
    second_face_score = data[0][0][7].flatten()

    # create dict from values
    parsed = {
        "dob": dob,
        "photo_taken": photo_taken,
        "full_path": full_path,
        "gender": gender,
        "name": name,
        "face_location": face_location,
        "face_score": face_score,
        "second_face_score": second_face_score,
    }
    full_data = pd.DataFrame(parsed)

    #
    face_present = full_data[np.isfinite(full_data["face_score"])]

    # make sure no bounding boxes are [1,1,1,1] which means no face to be found
    assert (
        len(face_present[face_present["face_location"].astype(str) == "[1 1 1 1]"]) == 0
    )

    # len(full_data[pd.isnull(full_data)["gender"]]) #2643 are genderless

    # dataframe with second faces removed:
    primary = face_present[pd.isnull(face_present["second_face_score"])]

    # get rid of cols don't need
    primary = primary.drop(["second_face_score"], axis=1).reset_index(drop=True)

    # add in birthdays
    birthday = primary["dob"].apply(datenum_to_datetime)
    age = primary["photo_taken"] - birthday
    primary["age"] = age
    primary = primary

    # assume need to be at least 1 year old
    primary = primary[primary.age >= 1]

    # add in age and remove humans outside age range
    primary = primary.drop(["dob"], axis=1)
    primary = primary[primary.age < 100].reset_index(drop=True)

    # add in bounds and save to csv
    bounds = pd.DataFrame(
        primary["face_location"].to_list(),
        columns=["bound0", "bound1", "bound2", "bound3"],
    )
    primary = pd.concat([primary, bounds], axis=1)  # .drop("face_location",axis=1)

    primary.to_csv(csv_path, index=False)


# function to parse datetime in matlab to python
def datenum_to_datetime(datenum, age=False):
    """
    source: https://gist.github.com/victorkristof/b9d794fe1ed12e708b9d
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    try:
        days = datenum % 1
        datetime_ = (
            datetime.fromordinal(int(datenum))
            + dt.timedelta(days=days)
            - dt.timedelta(days=366)
        )

        return datetime_.year  # , round((datetime.now() - datetime_).days / 365.25)
    except:
        return np.inf


def create_imdb_joint_csvs(root_dir="/home/jessekim"):
    csv_dir = os.path.join(root_dir, "data")
    logs_dir = os.path.join(root_dir, "logs")

    feature_extractor = ViTFeatureExtractor.from_pretrained(
        'google/vit-base-patch16-224-in21k'
    )

    # paths and load in matlab file of wikipedia image metadata

    wiki_data_dir = os.path.join(csv_dir, "wiki_crop")
    imdb_data_dir = os.path.join(csv_dir, "imdb_crop")

    wiki_csv_path = os.path.join(csv_dir, "face_age_processed_data.csv")
    imdb_csv_path = os.path.join(csv_dir, "imdb_face_age_processed_data.csv")

    generate_df_from_mat(
        wiki_data_dir, csv_dir, logs_dir, wiki_csv_path, feature_extractor, "wiki"
    )
    generate_df_from_mat(
        imdb_data_dir, csv_dir, logs_dir, imdb_csv_path, feature_extractor, "imdb"
    )

    ### create final wiki_imdb train_test splits
    wiki = pd.read_csv(wiki_csv_path)
    imdb = pd.read_csv(imdb_csv_path)

    wiki["full_path"] = "wiki_crop/" + wiki["full_path"]
    imdb["full_path"] = "imdb_crop/" + imdb["full_path"]

    imdb_wiki = pd.concat([wiki, imdb]).reset_index(drop=True)

    shuffled = imdb_wiki.sample(frac=1, random_state=42).reset_index(drop=True)
    shuffled = shuffled.drop(
        ["photo_taken", "gender", "name", "face_location", "face_score"], axis=1
    )

    # make train/test splits
    valid_idx = int(len(imdb_wiki) * 0.05)
    test_idx = int(len(imdb_wiki) * 0.15)

    valid = shuffled.iloc[:valid_idx]
    test = shuffled.iloc[valid_idx : valid_idx + test_idx]
    train = shuffled.iloc[valid_idx + test_idx :]
    print(valid.shape, test.shape, train.shape)

    valid.to_csv(os.path.join(csv_dir, "imdb_wiki_valid.csv"), index=False)
    test.to_csv(os.path.join(csv_dir, "imdb_wiki_test.csv"), index=False)
    train.to_csv(os.path.join(csv_dir, "imdb_wiki_train.csv"), index=False)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        help="where the experiment data is (directory)",
        default='/home/jessekim',
    )
    args = parser.parse_args(sys.argv[1:])

    print(args)

    create_imdb_joint_csvs(**args.__dict__)  # csv_dir=csv_dir, logs_dir=logs_dir)
