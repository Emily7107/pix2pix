import os
import json
from datetime import datetime
import torch
from torch import nn
import wandb
from PIL import Image
import numpy as np

def initialize_weights(layer):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif isinstance(layer, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0.0)
    return None

class Logger():
    def __init__(self,
                 exp_name: str='./runs',
                 filename: str=None):
        self.exp_name=exp_name
        self.cache={}
        if not os.path.exists(exp_name):
            os.makedirs(exp_name, exist_ok=True)
        self.date=datetime.today().strftime("%B_%d_%Y_%I_%M%p")
        if filename is None:
            self.filename=self.date
        else:
            self.filename="_".join([self.date, filename])
        fpath = f"{self.exp_name}/{self.filename}.json"
        with open(fpath, 'w') as f:
            data = json.dumps(self.cache)
            f.write(data)
            
        wandb.init(project="pix2pix_kitti", name="training_run_masking")
        
    def add_scalar(self, key: str, value: float, t: int):
        wandb.log({key: value}, step=t)
        return None
    
    def save_weights(self, state_dict, model_name: str='model'):
        fpath = f"{self.exp_name}/{model_name}.pt"
        torch.save(state_dict, fpath)
        artifact=wandb.Artifact('pix2pix_kitti',type='model')
        artifact.add_file(fpath)
        wandb.log_artifact(artifact)
        return None
    
    def update(self,):
        fpath = f"{self.exp_name}/{self.filename}.json"
        with open(fpath, 'w') as f:
            data = json.dumps(self.cache)
            f.write(data)
        return None
    
    def close(self,):
        fpath = f"{self.exp_name}/{self.filename}.json"
        with open(fpath, 'w') as f:
            data = json.dumps(self.cache)
            f.write(data)
        self.cache={}
        wandb.finish()
        return None
