import os
import torch
import pandas as pd
import torchvision.io as tvio


class XRaySet(torch.utils.data.Dataset):
    def __init__(self, csv_filepath, img_dir):
        self.img_labels = pd.read_csv(csv_filepath)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_data = self.img_labels.iloc[index]
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index, 2], '.jpeg')
        img = tvio.read_image(img_path, tvio.ImageReadMode.GRAY)
        label = img_data ['label']
        return img, label

X = XRaySet('chest_xray_data.csv', '../chest_xray')
print(X.__len__())