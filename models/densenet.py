import pandas as pd
import numpy as np
# from datetime import datetime 
# import datetime as dt
from torchvision import transforms, utils
import torchvision.models as models
import torch
import cv2
from PIL import Image
import torch.nn as nn
import os

class DenseNet(torch.nn.Module):
    def __init__(self, categories, device="cuda", output_type="classification", log=True, dropout=.40):
        super(DenseNet, self).__init__()
        self.CATEGORIES = categories
        self.device = device
        self.output_type = output_type
        self.log = log
                
        DENSENET_CLASSIFIER_IN = 2208
        self.model = models.densenet161(pretrained=True)
        #alter to take output of model and make into softmax 
        
        #if classifier:
        if output_type == "classification":
            print("model is a classifier")
            if self.log:
                self.model.classifier = nn.Sequential(
                    nn.Dropout(p=dropout, inplace=False), #DENSENET_NLL has 0.4 dropout...
                    nn.Linear(DENSENET_CLASSIFIER_IN, self.CATEGORIES),
                    nn.LogSoftmax())
            else:
                self.model.classifier = nn.Sequential(
                    nn.Dropout(p=dropout, inplace=False),
                    nn.Linear(DENSENET_CLASSIFIER_IN, self.CATEGORIES),
                    nn.Softmax())
                
            ### NEED TO CHANGE TO LOG SOFTAMX FOR NLL!!!
            
        elif output_type == "regression":
            print("model is regressor")
            #if regression
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=False),
                nn.Linear(DENSENET_CLASSIFIER_IN, 1),
                nn.Softplus())
        else:
            print("wrong input for net type")

    def forward(self, x):
        if self.output_type == "classification":
            softmaxes = self.model(x)

            classes = torch.tensor(np.arange(1,self.CATEGORIES + 1).reshape(-1, 1)).float().to(self.device)
            exp_age = torch.matmul(softmaxes, classes)
            
            
            return softmaxes, exp_age
        
        elif self.output_type == "regression":
            return self.model(x)
        
        else:
            return 
        