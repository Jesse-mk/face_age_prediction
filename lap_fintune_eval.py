import os 
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, "..")
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from main import load_in_loss, train
from utils import load_in_model, convert_types, setup_exp, log_experiment
from dataset_utils import create_datasets
from main import run_one_epoch
import argparse


os.environ["CUDA_VISIBLE_DEVICES"] = "5"


def load_in_mask_info(exp_name):
    mask_info = None
    if "1CHANNEL" in exp_name:
        mask_info = "1CHANNEL"
    elif "5CHANNEL" in exp_name:
        mask_info = "5CHANNEL"
    elif "BOX" in exp_name:
        mask_info = "BOX"
    else:
        mask_info = None
        
    if "premask" in exp_name:
        pre_mask = 1
    else:
        pre_mask = 0
    return mask_info, pre_mask
    
def create_lap_eval_dict(data_dir, exp_name, finetune=False):
    log_dir = os.path.join(data_dir, "logs")
    model_dir = os.path.join(data_dir, "models")
    
    #get tracker with all hyperparamters:
    try:
        tracker_csv_path = os.path.join(log_dir, "exp_tracker.csv")
        tracker = pd.read_csv(tracker_csv_path)
        args_dict = tracker[tracker["exp_name"] == exp_name].iloc[0]#.to_dict()

    except:
        print("lap")
        tracker_csv_path = os.path.join(log_dir, "lap_exp_tracker.csv")
        tracker = pd.read_csv(tracker_csv_path)
        args_dict = tracker[tracker["exp_name"] == exp_name].iloc[0]#.to_dict()

    
    #get model from previously trained experiment
    load_model = os.path.join(model_dir, exp_name + ".pt")
    
    #construction of args dict to make sure similar hyperparameters (OTHER THAN LR AND DROPOUT)
    args_dict = args_dict.drop(["date"])
    
    #for fine tuning set it small
    args_dict["lr"] = 1e-5
    args_dict["dropout"] = 0.33

    #add on eval script at front
    args_dict["date"] = datetime.now().date().strftime("%D")
    #load in mask info based on name
    args_dict["mask_info"], args_dict["pre_mask"] = load_in_mask_info(exp_name)

    #get lap dataset evaluation
    args_dict["data_source"] = "lap"
    if finetune:
        args_dict["exp_name"] = "LAP_finetune_" + args_dict["exp_name"] 
    else:
        args_dict["exp_name"] = "LAP_eval_" + args_dict["exp_name"] 
        
    #load in model
    args_dict["load_model"] = load_model
    args_dict["transform"] = None
    args_dict["bs"] = int(args_dict["bs"])
    
    #newer experiments have data_dir not model_dir, but just in case.
    try:
        data_dir = args_dict["model_dir"].split("models")[0]
        args_dict = args_dict.drop(["log_csv_path", "logs_dir", "model_dir"])
        args_dict["data_dir"] = data_dir
    except:
        pass
    
    return args_dict
    
def eval_model_lap(exp_name, data_dir="/home/jessekim"):
    #get similar args
    args = create_lap_eval_dict(data_dir=data_dir, exp_name=exp_name, finetune=False)

    #get the model
    model = load_in_model(
    args.model_type, args.output_type, args.loss_type, args.dropout, args.mask_info, args.pre_mask, args.load_model
    )
    
    # add to device (GPU vs CPU) and parallelize it
    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs')
        model = nn.DataParallel(model)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[], output_device=None)
    model = model.to(args.device)
    
    #get loss
    loss = load_in_loss(args.loss_type)
    
    #get test loader
    test_loader = create_datasets(
        dataset_type="test",
        data_dir=args.data_dir,
        transform=args.transform,
        mask_info=args.mask_info,
        bs=int(args.bs),
        shuffle=True,
        num_workers=8,
        data_source=args.data_source,
    )

    model, test_loss_vals, test_mae_vals, test_acc = run_one_epoch(
        model,
        test_loader,
        optimizer=None,
        mask_info=args.mask_info,
        scheduler=None,
        loss=loss,
        train=False,
        device=args.device,
        output_type=args.output_type,
        loss_type=args.loss_type,
    )
    print(test_loss_vals, test_mae_vals, test_acc)


def finetune_on_lap(exp_name, data_dir = "/home/jessekim"):
    """After training on wiki or imdb_wiki, fine tune on LAP dataset"""
    
    #get args dict from original experiment
    print(f"fine tuning on {exp_name}")
    args = create_lap_eval_dict(data_dir=data_dir, exp_name=exp_name, finetune=True)
    print(args)
    
    logs_dir = os.path.join(args["data_dir"], "logs")
    log_csv_path = os.path.join(logs_dir, "lap_exp_tracker.csv")

    # logging:
    setup_exp(log_csv_path, logs_dir, args["exp_name"])
    # add experiment to experiment tracker:
    log_experiment(log_csv_path, args)
    
    train(**args)
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", help="experiment to either finetune or evaluate")
    parser.add_argument("--finetune", help="whether to 'eval' (0) or 'finetune' (1)", default=0, type=int, choices= [1, 0])
    parser.add_argument("--data_dir", help="where age experiment directory is located", default="/home/jessekim")
    
    args = parser.parse_args(sys.argv[1:]).__dict__
    args["finetune"] = bool(args["finetune"])
    print(args["finetune"])
    print(type(args["finetune"]))
    
    if args["finetune"]:
        print("finetuning")
        finetune_on_lap(exp_name=args["exp_name"], data_dir=args["data_dir"])
    else:
        print("evaluating")
        eval_model_lap(exp_name=args["exp_name"], data_dir=args["data_dir"])

    
    
