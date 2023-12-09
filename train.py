import torchvision.transforms.v2 as T
from torch.utils.data.dataloader import DataLoader
import wandb
import torch
import torch.nn as nn
from torch import optim
import argparse
#custom library
import model
import utils

def train():
    wandb.login()

    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, help="data location", default='./data')
    parser.add_argument('--lr', type=float, help="learning rate", default=0.0002)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    args = parser.parse_args()

    wandb.init(
        project="face2comic",
        name=f"pix2pix-merge",
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
    try:
        model_gen.load_state_dict(torch.load('model/499-0.0001-step.pt')['gen'])
        model_dis.load_state_dict(torch.load('model/499-0.0001-step.pt')['dis'])
    except:
        model_gen.apply(model.initialize_weights)
        model_dis.apply(model.initialize_weights)

    criterion_gen = nn.BCELoss()
    criterion_dis = nn.L1Loss()

    lambda_pixel = 100

    patch = (1, 30, 30)
    optimizer_gen = optim.Adam(model_gen.parameters(), lr=config.lr, betas=config.beta)
    optimizer_dis = optim.Adam(model_dis.parameters(), lr=config.lr, betas=config.beta)

    model_gen.train()
    model_dis.train()
    model_gen, model_dis = model_gen.to(device), model_dis.to(device)

    for epoch in range(config.epoch):
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

            d_loss = (real_loss + fake_loss)
            d_loss.backward()
            optimizer_dis.step()

        #scheduler_dis.step()
        #scheduler_gen.step()
        wandb.log({'fake':wandb.Image(fake_comic[0].to('cpu')),
                   'face':wandb.Image(face[0].to('cpu')),
                   'comic':wandb.Image(comic[0].to('cpu'))})
        if epoch%10==0:
            torch.save({'gen':model_gen.state_dict(),
                        'dis':model_dis.state_dict()},
                        f'./model/{epoch}-merge.pt')
        
    wandb.finish()

if __name__ == "__main__":
    train()