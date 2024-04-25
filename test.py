import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
from progress.bar import IncrementalBar
from PIL import Image
import wandb
import numpy as np

from dataset import TryData
from dataset import transforms as T
from gan.generator import UnetGenerator
from gan.discriminator import ConditionalDiscriminator
from gan.utils import Logger, initialize_weights

parser = argparse.ArgumentParser(prog='top', description='Train Pix2Pix')
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--dataset", type=str, default="facades", help="Name of the dataset: ['facades', 'maps', 'cityscapes','trydata']")
parser.add_argument("--batch_size", type=int, default=1, help="Size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="Adams learning rate")
args = parser.parse_args()

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

transforms = T.Compose([T.Resize((256,256)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])])

print('Loading models!')
run=wandb.init(project="pix2pix", name="testing_run")
artifact = run.use_artifact('tyw7107/pix2pix/pix2pix:v28', type='model')
artifact_dir = artifact.download()
generator = UnetGenerator().to(device)
discriminator = ConditionalDiscriminator().to(device)

print(f'Loading "{args.dataset.upper()}" dataset!')
if args.dataset == 'trydata':
    dataset = TryData(root='.', transform=transforms, download=True, mode='val')

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
print('Start of evaluation process!')
generatedImage=[]
realImage=[]
inputImage=[]

with torch.no_grad():
    
    for x, real in dataloader:
        x = x.to(device)
        real = real.to(device)

        fake = generator(x)
        
        # move image to cpu
        fake_cpu = fake.cpu()
        fake_np = fake_cpu.squeeze(0).permute(1, 2, 0).numpy()
        generatedImage.append(fake_np)
        
        x_cpu = x.cpu().squeeze(0).permute(1, 2, 0).numpy()
        real_cpu = real.cpu().squeeze(0).permute(1, 2, 0).numpy()
        inputImage.append(x_cpu)
        realImage.append(real_cpu)
        
# change to numpy image 
generated_images_pil = [Image.fromarray(np.uint8(image * 255)) for image in generatedImage]
input_images_pil = [Image.fromarray(np.uint8(image * 255)) for image in inputImage]
real_images_pil = [Image.fromarray(np.uint8(image * 255)) for image in realImage]       

# upload to wandb
wandb.log({"Generated Image": [wandb.Image(image) for image in generated_images_pil]})
wandb.log({"Input Image": [wandb.Image(image) for image in input_images_pil]})
wandb.log({"Real Image": [wandb.Image(image) for image in real_images_pil]})
