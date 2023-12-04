from torch.utils.data import Dataset
from os.path import join
from os import listdir
import cv2

class CustomDataset(Dataset):
    def __init__(self, path2img, transform=None):
        super().__init__()
        self.path2real = join(path2img, 'real')
        self.path2comic = join(path2img, 'comic')
        self.img_filenames = [x for x in listdir(self.path2comic)]
        self.transform = transform

    def __getitem__(self, index):
        real = cv2.imread(join(self.path2real, self.img_filenames[index]))
        comic = cv2.imread(join(self.path2comic, self.img_filenames[index]))
        
        if self.transform:
            real = self.transform(real)
            comic = self.transform(comic)
        return real, comic

    def __len__(self):
        return len(self.img_filenames)