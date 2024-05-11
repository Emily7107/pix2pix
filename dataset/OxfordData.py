import glob
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from .utils import download_and_extract

class OxfordData(Dataset):
    def __init__(self,
                 root: str='.',
                 transform=None,
                 download: bool=False,
                 mode: str='train',
                 direction: str='A2B'):
        
        self.root=root
        self.filesA=sorted(glob.glob(f"{root}/OxfordData/origin/{mode}/*.png"))
        self.filesB=sorted(glob.glob(f"{root}/OxfordData/intensity/{mode}/*.png"))
        self.transform=transform
        self.download=download
        self.mode=mode
        self.direction=direction
        
    def __len__(self,):
        return max(len(self.filesA),len(self.filesB))
    
    def __getitem__(self, idx):
        
        imgA = Image.open(self.filesA[idx]).convert('RGB')
        imgB = Image.open(self.filesB[idx]).convert('RGB')
        
        if self.transform:
            imgA, imgB = self.transform(imgA, imgB)
            
        return imgA, imgB
        