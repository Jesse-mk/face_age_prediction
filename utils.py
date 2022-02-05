import sys
import argparse
import os
import datetime
import numpy as np
import torch
import sys
import imutils
import dlib
import pandas as pd
import datetime as dt
from datetime import datetime
import cv2
from torch.utils.data import Dataset, DataLoader
from imutils import face_utils
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn as nn

from models import AlteredVITModel, DenseNet, VisionTransformer

## CONSTANTS:
IMAGE_SHAPE = 224
COL_ORDER = [
    "date",
    "data_dir",
    "seed",
    "train_data",
    "exp_name",
    "bs",
    "shuffle",
    "patience_limit",
    "num_epochs",
    "lr",
    "num_workers",
    "device",
    "model_type",
    "load_model",
    "loss_type",
    "optimizer_type",
    "scheduler_type",
    "output_type",
    "data_source",
    "dropout",
    "pre_mask",
    "mask_info",
]
MAX_LABEL = 98

# logs_dir = "/home/jessekim/logs"
# ROOT = "/home/jessekim/"
# data_dir = os.path.join(ROOT, "data")
# logs_dir = os.path.join(ROOT, "logs")
# # path for jessekim_new account
# shape_predictor = os.path.join(ROOT,"models/shape_predictor_68_face_landmarks.dat")


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(shape_predictor)

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict(
    [
        ("mouth", (48, 68)),
        ("eyebrows", (17, 27)),
        ("eyes", (36, 48)),
        ("nose", (27, 35)),
        ("jaw", (0, 17)),
    ]
)

"""The facial detection functions are based off of those created by Adrian Rosebrock:
source: https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
source: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/"""


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    H, W, C = image.shape
    mask = np.zeros((H, W, 1))
    overlay = mask.copy()
    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(255), (255), (255), (255), (255), (255), (255)]

    mask_dict = {}
    for face_ids in FACIAL_LANDMARKS_IDXS:
        mask_dict[face_ids] = np.zeros((H, W, 1))
        # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]

        if name == "jaw":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(mask, ptA, ptB, colors[i], 25)
        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(mask, [hull], -1, colors[i], -1)
            cv2.drawContours(mask, [hull], -1, colors[i], 25)
        mask_dict[name] = mask
        mask = overlay.copy()
    # return the output image
    return mask_dict


def generate_masks(img_path, predictor, detector, regions=False, box=False):
    N_REGIONS = 5
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    H, W, C = image.shape
    mask = np.zeros((H, W, 1))

    if regions:
        mask_dict = {}
        for face_ids in FACIAL_LANDMARKS_IDXS:
            mask_dict[face_ids] = np.zeros((H, W, 1))

    out_viz = np.array([])
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        out_viz = visualize_facial_landmarks(image, shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)

        # rectangle box
        if box:
            mask = cv2.rectangle(mask, (x, y), (x + w, y + h), (255), -1)
            return mask

        if regions:
            pass
        else:
            for idx, (x, y) in enumerate(shape):
                if regions:
                    region = inv_regions[idx]
                    curr_mask = mask_dict[region]
                    cv2.circle(curr_mask, (x, y), 1, (255), 50)  # (0, 0, 255), 5)
                else:
                    cv2.circle(mask, (x, y), 1, (255), 5)  # (0, 0, 255), 5)

    if regions:
        try:
            return np.array(list(out_viz.values())).squeeze(-1).transpose(1, 2, 0)
        except:
            return np.array(np.zeros((H, W, 5)))
    else:
        return mask


def setup_exp(log_csv_path, logs_dir, exp_name):
    if os.path.exists(log_csv_path):
        pass
    else:
        pd.DataFrame(columns=COL_ORDER).to_csv(log_csv_path, index=False)

    curr_exp_path = os.path.join(logs_dir, exp_name + ".csv")

    # set up csv for specific experiment (make a new one each time):

    ind_cols = (
        "experiment_name",
        "train",
        "date",
        "epoch",
        "loss",
        "mae loss",
        "accuracy",
    )
    pd.DataFrame(columns=ind_cols).to_csv(curr_exp_path, index=False)


def log_output(exp_name, logs_dir, epoch, loss, mae_loss, acc, train):
    curr_exp_path = os.path.join(logs_dir, exp_name + ".csv")
    date = datetime.now().date().strftime("%D")
    with open(curr_exp_path, "a") as fd:
        fd.write(
            ",".join(
                [
                    exp_name,
                    train,
                    date,
                    str(epoch),
                    str(np.round(loss, 5)),
                    str(np.round(mae_loss, 5)),
                    str(np.round(acc, 5)),
                ]
            )
            + "\n"
        )


def log_experiment(log_csv_path, args_dict):
    print("logging experiment")
    with open(log_csv_path, "a") as fd:
        fd.write(",".join([str(args_dict[col]) for col in COL_ORDER]) + "\n")


def load_in_model(
    model_type, output_type, loss_type, dropout, mask_info, pre_mask, load_model=None
):
    CATEGORIES = 99
    log_loss = loss_type == "NLL"

    if model_type == "ViT":
        POOLED_SIZE = 768
        model = VisionTransformer(
            POOLED_SIZE,
            CATEGORIES,
            output_type=output_type,
            log=log_loss,
            dropout=dropout,
        )

    elif model_type == "DenseNet":
        model = DenseNet(
            CATEGORIES, output_type=output_type, log=log_loss, dropout=dropout
        )

    elif mask_info in ["1CHANNEL", "5CHANNEL", "BOX"]:
        POOLED_SIZE = 768
        in_channels = 1
        if mask_info == "5CHANNEL":
            print("5channel!")
            in_channels = 5

        model = AlteredVITModel(
            premodel=pre_mask,
            POOLED_SIZE=POOLED_SIZE,
            CATEGORIES=CATEGORIES,
            in_channels=in_channels,
            dropout=dropout,
        )

    else:
        print("model not listed, will throw error")

    # if pickup from checkpoint:
    if load_model:
        print(f"loaded from statedict at {load_model}")
        model.load_state_dict(torch.load(load_model))

    return model


def load_in_loss(loss_type):
    if loss_type == "NLL":
        print("NLL")
        loss = nn.NLLLoss()
    elif loss_type == "MAE":
        loss = torch.nn.L1Loss()
    elif loss_type == "MSE":
        loss = nn.MSELoss()
    elif loss_type == "CE":
        loss = nn.CrossEntropyLoss()
    else:
        print("not a correct loss, will throw error")
    return loss


def convert_types(dict_):
    for key in dict_.keys():
        try:
            if int(dict_[key]) == float(int(dict_[key])):
                dict_[key] = int(dict_[key])
        except:
            try:
                dict_[key] = float(dict_[key])
            except:
                pass
    return dict_


def parse_args(arguments):
    parser = argparse.ArgumentParser()

    # === Environment Paths ===
    parser.add_argument(
        "--data_dir",
        default="/home/jessekim",
        help="data dir where 'root', 'model' dir are located",
    )

    # === Environment config ===
    parser.add_argument("--seed", default=42, help="seed for reproducability")
    parser.add_argument(
        "--train_data",
        choices=["train", "test"],
        default="train",
        help="modes for the model",
    )

    # === Experiment config ===
    parser.add_argument(
        "--exp_name",
        default=datetime.now().strftime("experiment_%Y_%m_%d-%H_%M_%S"),
        help="Experiment name",
    )
    parser.add_argument("--bs", default=16, help="batch_size for training model. ")
    parser.add_argument(
        "--shuffle",
        choices=[True, False],
        default=True,
        help="batch_size for training model. ",
    )
    parser.add_argument(
        "--patience_limit",
        default=10,
        help="Number of epochs before validation increases to early stop at, 10",
    )
    parser.add_argument("--num_epochs", default=250, help="Number of epochs", type=int)
    parser.add_argument("--lr", default=0.001, help="Optimizer learning rate", type=int)
    parser.add_argument("--dropout", default=0.40, help="dropout rate", type=int)

    # === Dataloader config ===
    parser.add_argument(
        "--num_workers", default=4, help="Number of workers for the dataloader"
    )
    parser.add_argument("--device", default="cuda", help="cuda or not")

    # For data loading, passing pin_memory=True to a DataLoader will automatically put the fetched data
    # Tensors in pinned memory, and thus enables faster data transfer to CUDA-enabled GPUs.
    #     parser.add_argument("--no-pin-memory", action="store_false", help="Don't pin memory for the dataloader")

    # === Model config ===
    parser.add_argument(
        "--model_type",
        choices=["ViT", "DenseNet", "AlteredVIT"],
        default="ViT",
        help="The model to use",
    )
    parser.add_argument(
        "--mask_info",
        choices=["1CHANNEL", "BOX", "5CHANNEL", "4CHANNELVIT", ""],
        default="BOX",
        help="The mask type to use",
    )
    parser.add_argument(
        "--transform", choices=[None], default=None, help="str name of trnasform to use"
    )

    parser.add_argument(
        "--data_source",
        choices=["wiki", "imdb_wiki"],
        default="wiki",
        help="The source dataset to use",
    )
    parser.add_argument(
        "--pre_mask", choices=[1, 0], default=0, help="The mask type to use", type=int,
    )
    parser.add_argument(
        "--load_model", default="", help="The path to load checkpoint from (str)"
    )
    parser.add_argument(
        "--loss_type",
        choices=["MSE", "MAE", "NLL", "CE"],
        default="MSE",
        help="The loss used in training.",
    )
    parser.add_argument(
        "--optimizer_type",
        choices=["adam", "sgd"],
        default="adam",
        help="The optimizer used in training.",
    )
    parser.add_argument(
        "--scheduler_type",
        choices=["linear", "cosine"],
        default="linear",
        help="The scheduler used in training",
    )
    parser.add_argument(
        "--output_type",
        choices=["regression", "classification"],
        default="regression",
        help="output type (reg or class",
    )

    return parser.parse_args(arguments)
