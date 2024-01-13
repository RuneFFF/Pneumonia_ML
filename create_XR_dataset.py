import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import os
import numpy as np
import pandas as pd
import cv2

# from tqdm import tqdm
# import skimage
# from skimage.transform import resize
# import keras



class XRaySet(Dataset):
    def __int__(self, root_dir, is_train):
        self.root_dir = root_dir
        self.is_train = is_train

    def __len__(self):
          return len(self.dataframe)

    def __getitem__(self, type, person, indx):

        if torch.is_tensor(indx):
            indx = indx.tolist()

        if type == 'n':
            img_name = os.path.join(self.root_dir, '/NORMAL', 'n_'+indx)
        else:
            if 'v' in type:
                img_name = os.path.join(self.root_dir, '/PNEUMONIA',
                                        'p_' + person + '_v_' + indx)
            else:
                img_name = os.path.join(self.root_dir, '/PNEUMONIA',
                                        'p_' + person + '_b_' + indx)

        pre_image = cv2.imread(img_name)
        image = Image.fromarray(pre_image)  #convert to PIL

        if self.is_train:
            label_key = self.type
            label = torch.tensor(int(label2id[label_key]))
        else:
            label = torch.tensor(1)

        return image, label

batch_size = 4

test_data = XRaySet(root_dir = 'chest_xray/test', is_train=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=2, shuffle=False)



# train_dir = "chest_xray/train/"
# test_dir = "chest_xray/test/"

# def get_data(folder):
#     X = []
#     y = []
#     for folderName in os.listdir(folder):
#         if not folderName.startswith('.'):
#             if folderName in ['NORMAL']:
#                 label = 0
#             elif folderName in ['PNEUMONIA']:
#                 label = 1
#             else:
#                 label = 2
#             for image_filename in tqdm(os.listdir(folder + folderName)):
#                 img_file = cv2.imread(folder + folderName + '/' + image_filename)
#                 if img_file is not None:
#                     img_file = skimage.transform.resize(img_file, (150, 150, 3))
#                     # img_file = scipy.misc.imresize(arr=img_file, size=(150, 150, 3))
#                     img_arr = np.asarray(img_file)
#                     X.append(img_arr)
#                     y.append(label)
#     X = np.asarray(X)
#     y = np.asarray(y)
#     return X, y
