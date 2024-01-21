import os
import numpy as np
from PIL import Image
import cv2

def downsample(path, width, height, savepath):
    img = cv2.imread(path, 0)
    old_height, old_width = np.shape(img)
    ratio = width/height


    if old_width/old_height > ratio:
        bar_size = round(((old_width / ratio) - old_height)/2)
        bar = np.zeros((bar_size, old_width))
        barred = np.row_stack((bar, img, bar))
    elif old_width/old_height < ratio:
        bar_size = round(((old_height * ratio) - old_width)/2)
        bar = np.zeros((old_height, bar_size))
        barred = np.column_stack((bar, img, bar))
    else:
        barred = img

    final = (cv2.resize(barred, dsize=(width, height), interpolation=cv2.INTER_CUBIC))
    #Image.fromarray(final).convert('L').save(savepath)
    Image.fromarray(final).save(savepath)

#downsample('IM-0073-0001.jpeg', 750, 500, 'img.jpeg')
for root,_,files in os.walk('../chest_xray'):
    for filename in files:
        if filename[-5:] == ('.jpeg'):
            downsample(os.path.join(root,filename), 750, 500, os.path.join(root,filename))