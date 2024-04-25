import torch
import matplotlib.pyplot as plt

def evaluate(val_d1,name,G):
    with torch.no_grad():
        fig,axes=plt.subplots(6,8,figsize=(12,12))