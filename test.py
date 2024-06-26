import torch
from torch.utils.data import DataLoader
import argparse
import wandb
import os

from dataset import TryData
from dataset import KITTIdata
from dataset import OxfordData
from dataset import transforms as T
from dataset.transforms import ToImage
from gan.generator import UnetGenerator

parser = argparse.ArgumentParser(prog='top', description='Test Pix2Pix')
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--dataset", type=str, default="trydata", help="Name of the dataset: ['facades', 'maps', 'cityscapes','trydata','oxforddata']")
parser.add_argument("--batch_size", type=int, default=1, help="Size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="Adams learning rate")
parser.add_argument("--version",type=int,default=30,help="Enter the version of model")
args = parser.parse_args()

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

transforms = T.Compose([T.CenterCrop((256,256)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])])

print('Loading models!')
run=wandb.init(project="pix2pix_oxford", name="testing_run")
path=f'tyw7107/pix2pix_oxford/pix2pix_kitti:v{args.version}'
artifact = run.use_artifact(path, type='model')
artifact_dir = artifact.download()
model_path = os.path.join(artifact_dir, 'generator.pt')
generator = UnetGenerator().to(device)
generator.load_state_dict(torch.load(model_path))

print(f'Loading "{args.dataset.upper()}" dataset!')
if args.dataset == 'trydata':
    dataset = TryData(root='.', transform=transforms, download=True, mode='test')
elif args.dataset == 'oxforddata':
    dataset = OxfordData(root='.', transform=transforms, download=True, mode='test')
else:
    dataset = KITTIdata(root='.', transform=transforms, download=True, mode='test')

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
print('Start of evaluation process!')
generatedImage=[]
realImage=[]
inputImage=[]

toImage=ToImage()

with torch.no_grad():
    
    for x, real in dataloader:
        x = x.to(device)
        real = real.to(device)

        fake = generator(x)
        
        fake_pil = toImage(fake.cpu())
        real_pil = toImage(real.cpu())
        input_pil = toImage(x.cpu())

        generatedImage.append(fake_pil)
        realImage.append(real_pil)
        inputImage.append(input_pil)
        
# upload to wandb
wandb.log({"Generated Image": [wandb.Image(image) for image in generatedImage]})
wandb.log({"Input Image": [wandb.Image(image) for image in inputImage]})
wandb.log({"Real Image": [wandb.Image(image) for image in realImage]})