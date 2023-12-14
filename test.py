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
    parser.add_argument('--model', type=str, help="model location", default='./model1.pt')
    parser.add_argument('-camera',type=int, help="camera number", default=0)
    args = parser.parse_args()
    device = 'cpu'
    model_gen = model.GeneratorUNet()
    model_gen.load_state_dict(torch.load(args.model,map_location='cpu')['gen'])
    model_gen.eval()
    model_gen = model_gen.to(device)

    data_transform = T.Compose([T.Resize(256),
                                T.CenterCrop((256,256)),
                                T.ToImage(),
                                T.ToDtype(torch.float32, scale=True),
                                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                ])
    
    cap = cv2.VideoCapture(args.camera)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,512*h/w)
    ret, real_img = cap.read()
    while(1):
        pil_img = Image.fromarray(cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB))
        # real_img = cv2.resize(real_img,(600,400),interpolation=cv2.INTER_AREA)
        img = data_transform(pil_img)
        img = img.unsqueeze(dim=0)
        img = img.to(device)
        with torch.no_grad():
            img = model_gen(img).to('cpu')

        img = utils.unnormalize(img.squeeze(dim=0))
        cv2.imshow('real',real_img)
        cv2.imshow('comic', cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        ret, real_img = cap.read()
        k = cv2.waitKey(10)#wait 10ms
        if k==27:
            break
cv2.destroyAllWindows()

if __name__ == "__main__":
    test()