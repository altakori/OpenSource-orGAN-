import torchvision.transforms.v2 as T
from torch.utils.data.dataloader import DataLoader
from os.path import join
from PIL import Image
import wandb
import torch
import torch.nn as nn
from torch import optim
import argparse
import cv2
import numpy as np
#custom library
import model
import utils

def test():

    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, help="data location", default='./data')
    args = parser.parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_gen = model.GeneratorUNet()
    model_gen.load_state_dict(torch.load('model/4.pt')['gen'])
    model_gen.eval()
    model_gen = model_gen.to(device)

    data_transform = T.Compose([T.Resize((256,256)),
                                T.ToImage(),
                                T.ToDtype(torch.float32, scale=True),
                                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                ])

    test_image = Image.open(join(f'{args.location}/faces', '0.jpg')).convert('RGB')
    test_image = data_transform(test_image)
    test_image = test_image.unsqueeze(dim=0)
    test_image = test_image.to(device)
    with torch.no_grad():
        test_image = model_gen(test_image).to('cpu')

    test_image = utils.unnormalize(test_image.squeeze(dim=0))
    print(np.array(test_image))
    cv2.imshow('real',cv2.imread(join(f'{args.location}/faces', '0.jpg')))
    cv2.imshow('comic', cv2.cvtColor(np.array(test_image), cv2.COLOR_RGB2BGR))
    cv2.waitKey()

if __name__ == "__main__":
    test()