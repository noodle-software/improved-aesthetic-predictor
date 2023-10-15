import webdataset as wds
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import io
import matplotlib.pyplot as plt
import os
import json
import argparse
from warnings import filterwarnings
from pathlib import Path

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
import tqdm

from os.path import join
from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json

import clip


from PIL import Image, ImageFile

# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class AestheticPredictor:
    def __init__(self, model, model2, preprocess, device):
        self.model = model
        self.model2 = model2
        self.preprocess = preprocess
        self.device = device

    def eval_image(self, pil_image):

        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model2.encode_image(image)

            im_emb_arr = normalized(image_features.cpu().detach().numpy() )
        
            prediction = self.model(torch.from_numpy(im_emb_arr).to(self.device).type(torch.cuda.FloatTensor))
            score = prediction.item()
            # metadata = PngInfo()
            # metadata.add_text("aesthetic_score", str(score))
            
            # #print(f"before: {pil_image.info}")
            # #print(f"before: {pil_image.text}")
            # pil_image.save(img_path, pnginfo=metadata)
            # after_save = Image.open(img_path)
            #print(f"after: {after_save.info}")
            #print(f"after: {after_save.text}")

            #print(f"{img_path} score:{score}")
            #print( prediction )
            #print(prediction.item())
            return score
        
    @staticmethod
    def from_config(config):
        model_name = config.get('model_name', "sac+logos+ava1-l14-linearMSE.pth")
        
        model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

        # load the model you trained previously or the model available in this repo
        s = torch.load(f"improved_aesthetic_predictor/{model_name}")   
    
        model.load_state_dict(s)
        
        model.to("cuda")
        model.eval()
    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64
        return AestheticPredictor(model, model2, preprocess, device)


def process_folder(input_folder, predictor):
    for file in Path(input_folder).glob('*.png'):
        img_path = str(file)
        print(f"eval_image {img_path}")
        pil_image = Image.open(img_path)
        predictor.eval_image(pil_image)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_folder', help='path to input images')
    parser.add_argument('-m', '--model_name', help='the name of the model to use; needs to be in the current directory',
                        default="sac+logos+ava1-l14-linearMSE.pth")
    args = parser.parse_args()
    
    #model, model2, preprocess, device = setup(args.model_name)
    predictor = AestheticPredictor.from_config(args.model_name)

    if args.input_folder:
        process_folder(args.input_folder, predictor)
    else:
        predictor.eval_image(Image.open("test_half-decent.png"))
        predictor.eval_image(Image.open("test_not-good.png"))
        predictor.eval_image(Image.open("test_bad.png"))
