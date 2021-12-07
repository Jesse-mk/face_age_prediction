from torchvision import transforms, utils
import torch.nn as nn
import pandas as pd
import numpy as np
# from datetime import datetime 
# import datetime as dt
# import torchvision.models as models
import torch
import cv2
from transformers import ViTConfig, ViTModel,  ViTFeatureExtractor
from PIL import Image
import torch.nn as nn
import os


class VisionTransformer(torch.nn.Module):
    def __init__(self, pooled_size, categories, device="cuda", output_type="classification", log=True, dropout=.40):
        super(VisionTransformer, self).__init__()
        self.POOLED_SIZE = pooled_size
        self.CATEGORIES = categories
        self.device = device
        self.output_type = output_type
        self.log = log
        
        self.model_ = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.HIDDEN_SIZE = self.model_.config.hidden_size
        
        self.classifier = nn.Sequential(
        nn.Linear(self.POOLED_SIZE, self.CATEGORIES),
        nn.LogSoftmax())
        
        #if classifier:
        if output_type == "classification":
            print("model is a classifier")
            
            if self.log:
                self.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=False),
                nn.Linear(self.POOLED_SIZE, self.CATEGORIES),
                nn.LogSoftmax())
            else:
                self.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=False), #VIT_NLL had no dropout
                nn.Linear(self.POOLED_SIZE, self.CATEGORIES),
                nn.Softmax())
            
        elif output_type == "regression":
            print("model is regressor")
            #if regression
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=False),
                nn.Linear(self.POOLED_SIZE, 1),
                nn.Softplus())
        else:
            print("wrong input for net type")


    def forward(self, x):
        if self.output_type == "classification":
            out = self.model_(x)
            #use last hidden layer
            softmaxes = self.classifier(out["last_hidden_state"][:,0,:])            

            classes = torch.tensor(np.arange(1,self.CATEGORIES + 1).reshape(-1, 1)).float().to(self.device)
            exp_age = torch.matmul(softmaxes, classes)
            return softmaxes, exp_age
        
        elif self.output_type == "regression":
            out = self.model_(x)
            #use last hidden layer
            softmaxes = self.classifier(out["last_hidden_state"][:,0,:])
            return softmaxes
            
        else:
            return 
    