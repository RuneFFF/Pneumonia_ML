import sys
import os
import numpy as np
from PIL import Image
import cv2

def downsample(path, width, height):
    img = cv2.imread(path)
    downsampled = Image.fromarray(cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)).convert("L")
    downsampled.save(path)

downsample('test_image.jpeg', 750, 500)
'''
for root,_,files in os.walk('../chest_xray'):
    for filename in files:
        if filename[-5:] == ('.jpeg'):
            downsample(os.path.join(root,filename), 750, 500)
'''