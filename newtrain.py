from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader
import wandb
import torch
import torch.nn as nn
from torch import optim
#from torch.optim.lr_scheduler import StepLR
import argparse
#import time
#custom library
import model
import utils

def train():
    wandb.login()

    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, help="data location", default='./data')
    parser.add_argument('--lr', type=float, help="learning rate", default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    args = parser.parse_args()

    wandb.init(
        project="face2comic",
        name=f"pix2pix-model-fix",
        config={
            "location":args.location,
            "lr":args.lr,
            "batch_size":args.batch_size,
            "num_workers":args.num_workers,
            "beta":(args.beta1, args.beta2),
            "epoch":args.epoch,
        }
    )

    config = wandb.config
    utils.all_seed(42)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    data = utils.CustomDataset(config.location)

    data_loader = DataLoader(data,
                            batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=config.num_workers,
                            pin_memory=True)

    model_gen = model.GeneratorUNet()
    model_dis = model.Discriminator()
    model_gen.apply(model.initialize_weights)
    model_dis.apply(model.initialize_weights)

    criterion_gen = nn.BCELoss()
    criterion_dis = nn.L1Loss()

    lambda_pixel = 100

    patch = (1, 32, 32)
    optimizer_gen = optim.Adam(model_gen.parameters(), lr=config.lr, betas=config.beta)
    optimizer_dis = optim.Adam(model_dis.parameters(), lr=config.lr, betas=config.beta)
    #scheduler_gen = StepLR(optimizer_gen, step_size=20, gamma=0.1)
    #scheduler_dis = StepLR(optimizer_dis, step_size=20, gamma=0.1)

    model_gen.train()
    model_dis.train()
    model_gen, model_dis = model_gen.to(device), model_dis.to(device)

    for epoch in range(config.epoch):
        #start = time.time()
        for face, comic in data_loader:
            size = comic.size(0)
            face, comic = face.to(device), comic.to(device)
            face_label = torch.ones(size, *patch, requires_grad=False).to(device)
            comic_label = torch.zeros(size, *patch, requires_grad=False).to(device)

            model_gen.zero_grad()
            
            fake_comic = model_gen(face).to(device)
            out_dis = model_dis(fake_comic, comic).to(device)

            gen_loss = criterion_gen(out_dis, face_label)
            pixel_loss = criterion_dis(fake_comic, comic)
            g_loss = gen_loss + lambda_pixel * pixel_loss
            g_loss.backward()
            optimizer_gen.step()

            model_dis.zero_grad()
            
            out_dis = model_dis(comic, face).to(device)
            real_loss = criterion_gen(out_dis, face_label)

            out_dis = model_dis(fake_comic.detach(), face).to(device)
            fake_loss = criterion_gen(out_dis, comic_label)

            d_loss = (real_loss + fake_loss) / 2.
            d_loss.backward()
            optimizer_dis.step()

        #scheduler_dis.step()
        #scheduler_gen.step()
        wandb.log({'fake':wandb.Image(fake_comic[0].to('cpu')),
                   'face':wandb.Image(face[0].to('cpu')),
                   'comic':wandb.Image(comic[0].to('cpu'))})

        torch.save({'gen':model_gen.state_dict(),
                    'dis':model_dis.state_dict()},
                    f'./model/{epoch}-0.001-bs-8.pt')
        
        #end = time.time()
        #print('epoch %d time: '%epoch, end='')
        #print(f"{end - start:.5f} sec")
        
        
    wandb.finish()

if __name__ == "__main__":
    train()