from torch.utils.data import Dataset
from PIL import Image

import numpy as np
import os 


class ANIMAL10N(Dataset):

    def __init__(self,
                 img_dir,
                 transform=False):

        self.img_dir = img_dir
        self.transform = transform
        self.filepaths = []
        self.labels = []

        for filename in sorted(os.listdir(img_dir)):
            self.labels.append(int(filename.split("_")[0]))
            self.filepaths.append(os.path.join(img_dir,filename))


    def __len__(self):
        return len(self.filepaths)


    def __getitem__(self, idx):

        img = Image.open(self.filepaths[idx]).convert('RGB')
        label = np.array(self.labels[idx])

        if self.transform:
            img = self.transform(img)

        return img, label 