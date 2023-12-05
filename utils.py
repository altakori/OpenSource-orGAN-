from torch.utils.data import Dataset
from os.path import join
from os import listdir
from PIL import Image
import torchvision.transforms.v2 as T
import torch
import cv2

class CustomDataset(Dataset):
    def __init__(self, path2img, transform=None):
        super().__init__()
        self.path2real = join(path2img, 'faces')
        self.path2comic = join(path2img, 'comics')
        self.img_filenames = [x for x in listdir(self.path2comic)]
        self.transform = transform

    def __getitem__(self, index):
        real = Image.open(join(self.path2real, self.img_filenames[index])).convert('RGB')
        comic = Image.open(join(self.path2comic, self.img_filenames[index])).convert('RGB')
        if self.transform:
            real = self.transform(real)
            comic = self.transform(comic)
        return real, comic

    def __len__(self):
        return len(self.img_filenames)
    

def unnormalize(image):
    image = (image*0.5 + 0.5) * 255
    a = T.compose([T.ToDtype(torch.int32, scale=True),
                    T.ToPILImage()
                    ])
    a(image)
    return a
