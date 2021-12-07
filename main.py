import os
import sys
import pandas as pd
import numpy as np
import scipy.io as so
import datetime as dt
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models
from transformers import get_linear_schedule_with_warmup
import torch
import cv2
from transformers import ViTConfig, ViTModel  # ViTFeatureExtractor,
from torch import optim
from torch.optim import Adam
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import ViTFeatureExtractor, ViTModel
import torch.nn as nn

from dataset import FaceAgeDataset
from models import AlteredVITModel, DenseNet, VisionTransformer
from utils import (
    generate_masks,
    setup_exp,
    log_output,
    log_experiment,
    load_in_model,
    convert_types,
    load_in_loss,
    parse_args,
)
from dataset_utils import create_datasets

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def run_one_epoch(
    model,
    dataloader,
    optimizer,
    mask_info,
    scheduler,
    loss,
    train=True,
    device="cuda",
    output_type="classification",
    loss_type=None,
):

    loss_vals = 0
    mae_vals = 0
    acc = 0
    total_count = 0
    total_batch_count = 0

    model = model.float()
    if train:
        model.train()
    else:
        model.eval()

    for _, (img, label, mask) in enumerate(dataloader):
        if train:
            optimizer.zero_grad()
        # pipe everything over to device
        img = img.to(device)
        label = label.to(device).float()
        mask = mask.to(device)
        mask = mask.permute(0, 3, 1, 2).float()

        if mask_info:

            img = torch.cat((mask, img), axis=1)

        # put through model and classifier
        if output_type == "classification":
            if mask_info:
                pred_age = model(img)
            else:
                pred_age, out = model(img)
            error = torch.mean(
                torch.abs(torch.argmax(pred_age, axis=1) - label.float())
            ).item()
            # change acc
            acc += torch.sum((torch.argmax(pred_age, axis=1) == label).float()).item()

            # loss:
            if loss_type == "NLL":
                label = label.long()
                output = loss(pred_age, label)
            else:
                output = loss(out, label.reshape(pred_age.shape[0], -1))

        elif output_type == "regression":
            pred_age = model(img) * 98  # .float()
            # get MAE
            error = nn.L1Loss(reduction="mean")(pred_age, label.float()).item()
            output = loss(pred_age, label)

        else:
            pass

        if train:

            # make sure optimizer grad is 0, then update grad based on backwards, then step to update weights
            output.backward()
            optimizer.step()

        mae_vals += error
        loss_vals += output.item()

        # accuracy per epoch
        total_count += len(label)
        total_batch_count += 1

    if train:
        if scheduler:  # if not none
            print("scheduler step")
            scheduler.step()

    # model and average loss and accuracy
    return (
        model,
        loss_vals / total_batch_count,
        mae_vals / total_batch_count,
        acc / total_count,
    )


def run_experiment(
    exp_name,
    model_type,
    data_loaders,
    data_dir="/home/jessekim",
    optimizer_type="adam",
    mask_info="",
    pre_mask=0,
    scheduler_info=None,
    loss_type="NLL",
    lr=0.001,
    num_epochs=10,
    patience_limit=5,
    seed=42,
    load_model=None,
    device="cuda",
    output_type="classification",
    data_source="wiki",
    dropout=0.40,
):

    logs_dir = os.path.join(data_dir, "logs")
    model_dir = os.path.join(data_dir, "models")

    ## logging and experiment tracking:
    CATEGORIES = 99

    # set seed
    torch.manual_seed(seed)

    # early stopping
    prev_loss = np.inf
    best_loss = np.inf
    patience_counter = 0

    # data:
    train_loader, valid_loader = data_loaders

    print(f"model type is {model_type}")

    model = load_in_model(
        model_type, output_type, loss_type, dropout, mask_info, pre_mask
    )

    # add to device (GPU vs CPU) and parallelize it
    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs')
        model = nn.DataParallel(model)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[], output_device=None)
    model = model.to(device)

    model.train()
    print(f"device is {device}")

    # optimizer and scheduler and loss
    if optimizer_type == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    else:
        print("not a correct optimizer, will throw error")

    if scheduler_info:
        if scheduler_info["type"] == "cosine":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=scheduler_info["num_warmup_steps"],
                num_training_steps=scheduler_info["num_training_steps"],
            )
    else:
        print("no scheduler implemented")

    # load in loss type
    loss = load_in_loss(loss_type)

    # train
    for epoch in np.arange(num_epochs):
        # train
        model, train_loss_vals, train_mae_vals, train_acc = run_one_epoch(
            model,
            train_loader,
            optimizer,
            mask_info,
            scheduler,
            loss,
            train=True,
            device=device,
            output_type=output_type,
            loss_type=loss_type,
        )

        # log each epoch:
        log_output(
            exp_name,
            logs_dir,
            epoch,
            train_loss_vals,
            train_mae_vals,
            train_acc,
            "train",
        )

        # validation
        model, valid_loss_vals, valid_mae_vals, valid_acc = run_one_epoch(
            model,
            valid_loader,
            optimizer,
            mask_info,
            scheduler,
            loss,
            train=False,
            device=device,
            output_type=output_type,
            loss_type=loss_type,
        )

        # log each epoch:
        log_output(
            exp_name,
            logs_dir,
            epoch,
            valid_loss_vals,
            valid_mae_vals,
            valid_acc,
            "valid",
        )

        # early stopping:
        if valid_mae_vals > prev_loss:
            patience_counter += 1
        elif valid_mae_vals == prev_loss:
            pass
        else:  # valid_mae_vals < prev_loss:
            patience_counter = 0

            # if mae is lower, save model to directory
            if valid_mae_vals < best_loss:
                model_path = os.path.join(model_dir, exp_name + ".pt")
                torch.save(model.state_dict(), model_path)

        if valid_mae_vals < best_loss:
            best_loss = valid_mae_vals

        # test to make sure doesn't reach limit
        if patience_counter >= patience_limit:
            print(
                f"patience reached {patience_counter} round with no improvement, stopped at epoch {epoch}"
            )
            return "training is done"

        # update accuracy
        prev_loss = valid_mae_vals

        print(
            f"after epoch {epoch}, the train {loss_type} is {train_loss_vals}, train mae loss is {train_mae_vals}, and accuracy is {train_acc}"
        )
        print(
            f"after epoch {epoch}, the valid {loss_type} is {valid_loss_vals}, valid mae loss is {valid_mae_vals}, and accuracy is {valid_acc}"
        )


def train(
    date=None,
    exp_name="DenseNet_classification",
    model_type="DenseNet",
    optimizer_type="adam",
    scheduler_type="linear",
    loss_type="MSE",
    device="cuda",
    data_source="wiki",
    data_dir="/home/jessekim",
    mask_info=None,
    pre_mask="0",
    train_data=True,
    bs=16,
    shuffle=True,
    num_workers=4,
    lr=0.0001,
    num_epochs=250,
    patience_limit=5,
    seed=42,
    output_type="regression",
    load_model=None,
    dropout=0.40,
    transform=None,
):

    print("training")

    # get data
    train_dataloader = create_datasets(
        dataset_type="train",
        data_dir=data_dir,
        transform=transform,
        mask_info=mask_info,
        bs=bs,
        shuffle=True,
        num_workers=num_workers,
        data_source=data_source,
    )
    valid_dataloader = create_datasets(
        dataset_type="valid",
        data_dir=data_dir,
        transform=transform,
        mask_info=mask_info,
        bs=bs,
        shuffle=True,
        num_workers=num_workers,
        data_source=data_source,
    )

    dataloaders = (train_dataloader, valid_dataloader)

    scheduler_info = {
        "type": "cosine",
        "num_warmup_steps": 50,
        "num_training_steps": 200,
    }
    # run experiments (UPDATE THIS)
    run_experiment(
        exp_name=exp_name,
        model_type=model_type,
        data_dir=data_dir,
        optimizer_type=optimizer_type,
        scheduler_info=scheduler_info,
        loss_type=loss_type,
        mask_info=mask_info,
        pre_mask=pre_mask,
        data_loaders=dataloaders,
        lr=lr,
        num_epochs=num_epochs,
        patience_limit=patience_limit,
        seed=seed,
        output_type=output_type,
        load_model=load_model,
        device=device,
        data_source=data_source,
        dropout=dropout,
    )


# def model_eval(experiment_name):

#     test_dataloader = create_datasets(create_datasets="test", mask_info=mask_info, bs=bs, shuffle=True, num_workers=num_workers,  data_source=data_source)

#     model, test_loss_vals, test_mae_vals, test_acc = run_one_epoch(model, test_loader, optimizer=None, mask_info, scheduler, loss, train=False, device=device, output_type=output_type, loss_type=loss_type)


if __name__ == "__main__":
    # parse args:
    args = parse_args(sys.argv[1:])
    arg_dict = args.__dict__
    arg_dict["date"] = datetime.now().date().strftime("%D")
    arg_dict = convert_types(arg_dict)
    print(arg_dict)

    logs_dir = os.path.join(arg_dict["data_dir"], "logs")
    log_csv_path = os.path.join(logs_dir, "exp_tracker.csv")

    # logging:
    setup_exp(log_csv_path, logs_dir, arg_dict["exp_name"])
    # add experiment to experiment tracker:
    log_experiment(log_csv_path, arg_dict)

    # train start
    train(**arg_dict)


### USEFUL COMMANDS ###

## NLL DenseNet gets 100% accuracy when trained on size 120 (able to overfit) (has .40 drop)
#  CUDA_VISIBLE_DEVICES=[0,1] python main.py --model_type="DenseNet" --optimizer_type="adam" --scheduler_type="linear" --loss_type="NLL" --train_data="train" --bs=16 --num_workers=4 --lr=0.0005 --num_epochs=250 --output_type="classification" --exp_name="DenseNet_NLL"

### NLL VIT
# CUDA_VISIBLE_DEVICES=[0,1] python main.py --model_type="ViT" --optimizer_type="sgd" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=16 --num_workers=4 --lr=0.001 --num_epochs=250 --output_type="classification" --exp_name="VIT_NLL_0.25drop"


# CUDA_VISIBLE_DEVICES=[0,1] python main.py --model_type="ViT" --optimizer_type="sgd" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=16 --num_workers=4 --lr=0.01 --num_epochs=250 --output_type="classification" --exp_name="VIT_NLL_0drop"

### WIKI EXPERIMENTS

# CUDA_VISIBLE_DEVICES=[0,1] python main.py --model_type="AlteredVIT" --mask_info="BOX" --optimizer_type="adam" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=16 --num_workers=4 --lr=0.001 --num_epochs=250 --output_type="classification" --pre_mask="0" --exp_name="VIT_NLL_BOX_postmask"

# CUDA_VISIBLE_DEVICES=[0,1] python main.py --model_type="AlteredVIT" --mask_info="BOX" --optimizer_type="adam" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=16 --num_workers=4 --lr=0.001 --num_epochs=250 --output_type="classification" --pre_mask="1" --exp_name="VIT_NLL_BOX_premask"

# CUDA_VISIBLE_DEVICES=[0,1] python main.py --model_type="AlteredVIT" --mask_info="1CHANNEL" --optimizer_type="adam" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=16 --num_workers=4 --lr=0.001 --num_epochs=250 --output_type="classification" --pre_mask="0" --exp_name="VIT_NLL_1CHANNEL_postmask"

# CUDA_VISIBLE_DEVICES=[0,1] python main.py --model_type="AlteredVIT" --mask_info="1CHANNEL" --optimizer_type="adam" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=16 --num_workers=1 --lr=0.001 --num_epochs=250 --output_type="classification" --pre_mask="1" --exp_name="VIT_NLL_1CHANNEL_premask"


# python main.py --model_type="AlteredVIT" --mask_info="5CHANNEL" --optimizer_type="adam" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=16 --num_workers=8 --lr=0.001 --num_epochs=250 --output_type="classification" --pre_mask="1" --exp_name="VIT_NLL_5CHANNEL_premask"

# python main.py --model_type="AlteredVIT" --mask_info="5CHANNEL" --optimizer_type="adam" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=16 --num_workers=8 --lr=0.001 --num_epochs=250 --output_type="classification" --pre_mask="0" --exp_name="VIT_NLL_5CHANNEL_postmask"

### WIKI BASELINES:
# python main.py --model_type="DenseNet" --optimizer_type="adam" --mask_info="" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=16 --num_workers=8 --lr=0.001 --num_epochs=250 --output_type="classification" --exp_name="DenseNet_NLL_NOMASK_0DROP"

# python main.py --model_type="ViT" --optimizer_type="adam" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=16 --num_workers=8 --lr=0.001 --num_epochs=250 --output_type="classification" --exp_name="ViT_NLL_NOMASK_0DROP"

## REGULARIZATION DROP BASELINES (0.4)
# python main.py --model_type="DenseNet" --optimizer_type="adam" --mask_info="" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=16 --num_workers=8 --lr=0.001 --num_epochs=250 --output_type="classification" --exp_name="DenseNet_NLL_NOMASK_40DROP"

# python main.py --model_type="ViT" --optimizer_type="adam" --mask_info="" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=16 --num_workers=8 --lr=0.001 --num_epochs=250 --output_type="classification" --exp_name="ViT_NLL_NOMASK_40DROP"

### IMDB WIKI BASELINES
# python main.py --model_type="DenseNet" --optimizer_type="adam" --mask_info="" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=16 --num_workers=8 --lr=0.001 --num_epochs=250 --output_type="classification" --exp_name="test"

# python main.py --model_type="DenseNet" --optimizer_type="adam" --mask_info="" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=16 --num_workers=8 --lr=0.001 --num_epochs=250 --output_type="classification" --exp_name="DenseNet_NLL_NOMASK_40DROP_IMDBWIKI"

# python main.py --model_type="ViT" --optimizer_type="adam" --mask_info="" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=16 --num_workers=8 --lr=0.001 --num_epochs=250 --output_type="classification" --exp_name="ViT_NLL_NOMASK_40DROP_IMBD_WIKI"


### IMDB-WIKI FINETUNING BASELINES
# python main.py --model_type="DenseNet" --optimizer_type="adam" --mask_info="" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=32 --num_workers=8 --lr=0.00001 --num_epochs=250 --output_type="classification" --exp_name="DenseNet_NLL_NOMASK_40DROP_IMDBWIKI_lr.00001"

# python main.py --model_type="ViT" --optimizer_type="adam" --mask_info="" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=32 --num_workers=8 --lr=0.00001 --num_epochs=250 --output_type="classification" --exp_name="ViT_NLL_NOMASK_40DROP_IMBD_WIKI_lr.00001"

#### MASK LOW LEARNING RATE
# CUDA_VISIBLE_DEVICES=[0,1] python main.py --model_type="AlteredVIT" --mask_info="1CHANNEL" --optimizer_type="adam" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=32 --num_workers=4 --lr=0.000001 --num_epochs=250 --output_type="classification" --pre_mask="0" --exp_name="VIT_NLL_1CHANNEL_40DROP_postmask_lr=0.000001" --data_source="wiki"

# CUDA_VISIBLE_DEVICES=[0,1] python main.py --model_type="AlteredVIT" --mask_info="1CHANNEL" --optimizer_type="adam" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=32 --num_workers=4 --lr=0.000001 --num_epochs=250 --output_type="classification" --pre_mask="0" --exp_name="VIT_NLL_1CHANNEL_60DROP_postmask_lr=0.000001" --data_source="wiki"

### REVISIONS TO TOP:
# python main.py --model_type="ViT" --optimizer_type="adam" --mask_info="" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=32 --num_workers=16 --lr=0.001 --num_epochs=250 --output_type="classification" --exp_name="ViT_NLL_NOMASK_80DROP" --data_source="wiki" --dropout=0.80

# python main.py --model_type="ViT" --optimizer_type="adam" --mask_info="" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=32 --num_workers=16 --lr=0.0001 --num_epochs=250 --output_type="classification" --exp_name="ViT_NLL_NOMASK_40DROP_lr.0001" --data_source="wiki" --dropout=0.40

#  python main.py --model_type="ViT" --optimizer_type="adam" --mask_info="" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=32 --num_workers=16 --lr=0.01 --num_epochs=250 --output_type="classification" --exp_name="ViT_NLL_NOMASK_40DROP_lr.01" --data_source="wiki" --dropout=0.40

#  python main.py --model_type="ViT" --optimizer_type="sgd" --mask_info="" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=32 --num_workers=16 --lr=0.001 --num_epochs=250 --output_type="classification" --exp_name="ViT_NLL_NOMASK_40DROP_sgd_lr.001" --data_source="wiki" --dropout=0.40

### try to improve mask results:

# python main.py --model_type="AlteredVIT" --mask_info="1CHANNEL" --optimizer_type="adam" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=16 --num_workers=4 --lr=0.001 --num_epochs=250 --output_type="classification" --pre_mask="0" --exp_name="VIT_NLL_1CHANNEL_postmask0.75"  --data_source="wiki" --dropout=0.75

# python main.py --model_type="AlteredVIT" --mask_info="1CHANNEL" --optimizer_type="adam" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=32 --num_workers=4 --lr=0.001 --num_epochs=250 --output_type="classification" --pre_mask="1" --exp_name="VIT_NLL_1CHANNEL_premask0.75"  --data_source="wiki" --dropout=0.75

# python main.py --model_type="AlteredVIT" --mask_info="1CHANNEL" --optimizer_type="adam" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=48 --num_workers=4 --lr=0.0001 --num_epochs=250 --output_type="classification" --pre_mask="1" --exp_name="VIT_NLL_1CHANNEL_premask0.6_lr0001"  --data_source="wiki" --dropout=0.6

# python main.py --model_type="AlteredVIT" --mask_info="1CHANNEL" --optimizer_type="adam" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=16 --num_workers=4 --lr=0.001 --num_epochs=250 --output_type="classification" --pre_mask="0" --exp_name="VIT_NLL_1CHANNEL_postmaskIMDB_WIKI_40"  --data_source="imdb_wiki" --dropout=0.40

# python main.py --model_type="AlteredVIT" --mask_info="1CHANNEL" --optimizer_type="adam" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=16 --num_workers=4 --lr=0.001 --num_epochs=250 --output_type="classification" --pre_mask="0" --exp_name="ignore"  --data_source="wiki" --dropout=0.40
