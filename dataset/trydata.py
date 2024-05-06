import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .utils import download_and_extract

class TryData(Dataset):
    def __init__(self,
                 root: str='.',
                 transform=None,
                 download: bool=False,
                 mode: str='train',
                 direction: str='A2B'):
        
        self.root=root
        self.files=sorted(glob.glob(f"{root}/trydata/{mode}/*.png"))
        self.transform=transform
        self.download=download
        self.mode=mode
        self.direction=direction
        
    def __len__(self,):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        img = np.array(img)
        img = img /256
        img = Image.fromarray(img)
        img = img.convert('RGB')
        
        W, H = img.size
        cW = W//2
        imgA = img.crop((0, 0, cW, H))
        imgB = img.crop((cW, 0, W, H))
        
        if self.transform:
            imgA, imgB = self.transform(imgA, imgB)
            
        if self.direction == 'A2B':
            return imgA, imgB
        else:
            return imgB, imgA