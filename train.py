import torch
from torch import nn
from torch.utils.data import DataLoader
import time
import argparse
from progress.bar import IncrementalBar

from dataset import TryData
from dataset import KITTIdata
from dataset import OxfordData
from dataset import transforms as T
from gan.generator import UnetGenerator
from gan.discriminator import ConditionalDiscriminator
from gan.criterion import GeneratorLoss, DiscriminatorLoss
from gan.utils import Logger, initialize_weights

parser = argparse.ArgumentParser(prog='top', description='Train Pix2Pix')
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--dataset", type=str, default="kittidata", help="Name of the dataset: ['facades', 'maps', 'cityscapes','trydata','kittidata','oxforddata']")
parser.add_argument("--batch_size", type=int, default=1, help="Size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="Adams learning rate")
args = parser.parse_args()

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

transforms = T.Compose([T.Resize((256,256)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),])

print('Defining models!')
generator = UnetGenerator().to(device)
discriminator = ConditionalDiscriminator().to(device)

g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

g_criterion = GeneratorLoss(alpha=100)
d_criterion = DiscriminatorLoss()

print(f'Loading "{args.dataset.upper()}" dataset!')
if args.dataset == 'trydata':
    dataset = TryData(root='.', transform=transforms, download=True, mode='train')
elif args.dataset == 'oxforddata':
    dataset = OxfordData(root='.', transform=transforms, download=True, mode='train')
else:
    dataset = KITTIdata(root='.', transform=transforms, download=True, mode='train')

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
print('Start of training process!')
logger = Logger(filename=args.dataset)

for epoch in range(args.epochs):
    ge_loss = 0.
    de_loss = 0.
    start = time.time()
    bar = IncrementalBar(f'[Epoch {epoch+1}/{args.epochs}]', max=len(dataloader))

    for x, real in dataloader:
        x = x.to(device)
        real = real.to(device)

        fake = generator(x)
        fake_pred = discriminator(fake, x)
        
        real_copy = real
        mask0 = torch.zeros_like(real_copy)
        mask1 = torch.ones_like(real_copy)
        mask = torch.where(real_copy > 0, mask1,mask0)
        fake_mask = torch.mul(fake,mask)
        
        g_loss = g_criterion(fake, real, fake_pred)

        fake = generator(x).detach()
        fake_pred = discriminator(fake, x)
        real_pred = discriminator(real, x)
        
        d_loss = d_criterion(fake_pred, real_pred)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        ge_loss += g_loss.item()
        de_loss += d_loss.item()
        bar.next()

    bar.finish()

    g_loss = ge_loss / len(dataloader)
    d_loss = de_loss / len(dataloader)
    end = time.time()
    tm = (end - start)

    logger.add_scalar('generator_loss', g_loss, epoch+1)
    logger.add_scalar('discriminator_loss', d_loss, epoch+1)

    print("[Epoch %d/%d] [G loss: %.3f] [D loss: %.3f] ETA: %.3fs" % (epoch+1, args.epochs, g_loss, d_loss, tm))

logger.save_weights(generator.state_dict(), 'generator')
# logger.save_weights(discriminator.state_dict(), 'discriminator')
logger.close()

print('End of training process!')
