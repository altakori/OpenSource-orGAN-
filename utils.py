from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
import torch
from os.path import join
from os import listdir
from PIL import Image
import numpy as np
import random
import cv2

class CustomDataset(Dataset):
    def __init__(self, path2img, transform=None):
        super().__init__()
        self.path2real = join(path2img, 'faces')
        self.path2comic = join(path2img, 'comics')
        self.img_filenames = [x for x in listdir(self.path2comic)]
        self.transform = T.Compose([T.Resize((256,256)),
                                    T.ToImage(),
                                    T.ToDtype(torch.float32, scale=True),
                                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                    ])
        
    def custom_transform(self,real, comic):
        # resize = T.Resize((256,256))
        # real = resize(real)
        # comic = resize(comic)
        # i, j, h, w = T.RandomCrop.get_params(real, output_size=(256, 256))
        # real = T.functional.crop(real, i, j, h, w)
        # comic = T.functionalcrop(comic, i, j, h, w)
        if random.random() > 0.5:
            angle = random.uniform(-5,5)
            real = T.functional.rotate(real, angle)
            comic = T.functional.rotate(comic, angle)

        if random.random() > 0.5:
            real = T.functional.horizontal_flip(real)
            comic = T.functional.horizontal_flip(comic)

        return real, comic

    def __getitem__(self, index):
        real = Image.open(join(self.path2real, self.img_filenames[index])).convert('RGB')
        comic = Image.open(join(self.path2comic, self.img_filenames[index])).convert('RGB')
        #real = cv2.imread(join(self.path2real, self.img_filenames[index]))
        #comic = cv2.imread(join(self.path2comic, self.img_filenames[index]))
        real, comic = self.custom_transform(real, comic)
        real = self.transform(real)
        comic = self.transform(comic)
        return real, comic

    def __len__(self):
        return len(self.img_filenames)
    

def unnormalize(image):
    image = (image*0.5 + 0.5)
    a = T.Compose([T.ToPILImage(),
                    ])
    return a(image)

def all_seed(num):
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.backends.cudnn.benchmark=True