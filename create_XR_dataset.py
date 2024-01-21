import os
import torch
import pandas as pd
import torchvision.io as tvio
from skimage import io


class XRaySet(torch.utils.data.Dataset):
    def __init__(self, csv_filepath, root_dir, transform=None):
        self.img_labels = pd.read_csv(csv_filepath)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_data = self.img_labels.iloc[index]
        img_path = os.path.join(self.root_dir, self.img_labels.iloc[index, 1] + '.jpeg')
        img = io.imread(img_path)
        #img = tvio.read_image(img_path, tvio.ImageReadMode.GRAY)

        if self.transform:  #transforms image to torch tensor
            img = self.transform(img)

        #label = img_data['label']
        if img_data['label']=='N':
            label = 0
        elif img_data['label']=='B':
            label = 1
        elif img_data['label']=='V':
            label = 2

        label = torch.tensor(label, dtype=torch.float32)
        return img, label

X = XRaySet('chest_xray_data.csv', '../chest_xray')
print(X.__getitem__(3))