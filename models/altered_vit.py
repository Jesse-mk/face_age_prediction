import pandas as pd
import numpy as np
# from datetime import datetime 
# import datetime as dt
from torchvision import transforms, utils
# import torchvision.models as models
import torch
import cv2
import os
from transformers import ViTConfig, ViTModel,  ViTFeatureExtractor
from PIL import Image
import torch.nn as nn

class AlteredVITModel(torch.nn.Module):
    def __init__(self, premodel=False, POOLED_SIZE=768, CATEGORIES=98, in_channels=1,  dropout=.40):
        super(AlteredVITModel, self).__init__()
        self.POOLED_SIZE = POOLED_SIZE
        self.CATEGORIES = CATEGORIES
        self.premodel = premodel
        self.ORIGINAL_IN_CHANNELS = 3
        
        if premodel:
            print("pre modeling is used")
            self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            self.conv43 = nn.Conv2d(in_channels = self.ORIGINAL_IN_CHANNELS + in_channels, out_channels = 3, kernel_size = 1).float()
           
        else:
            print("altered input size used")
            vit_config = ViTConfig(hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout, num_channels=4)
            vit = ViTModel(config=vit_config)
            #load from pre_trained after config = change to pretrained config... so need to do something else
            # vit.from_pretrained('google/vit-base-patch16-224-in21k')

            #get pretrained state dict and add to model as new channel
            state_dict_16ViTModel = vit.from_pretrained('google/vit-base-patch16-224-in21k').state_dict()

            with torch.no_grad():
                extra_weights = nn.Conv2d(in_channels = in_channels, out_channels = 768, kernel_size=16).weight
                new_layer = np.concatenate((state_dict_16ViTModel["embeddings.patch_embeddings.projection.weight"], extra_weights), axis=1)
                state_dict_16ViTModel["embeddings.patch_embeddings.projection.weight"] = torch.tensor(new_layer)

            vit.load_state_dict(state_dict_16ViTModel)
            self.model = vit        
        
        #either way need a classifier:
        self.classifier = nn.Sequential(
            nn.Linear(self.POOLED_SIZE, self.CATEGORIES),
            nn.LogSoftmax())
        
        
    def forward(self, input):
        if self.premodel:
            out = self.conv43(input)
            out = self.model(out)
            out = self.classifier(out["pooler_output"])
        else:
            out = self.model(input)
            out = self.classifier(out["pooler_output"])
            
#         print(out.shape)
        return out    