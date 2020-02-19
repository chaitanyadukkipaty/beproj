import os
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.prepro import *
import cv2
import imageio
import numpy as np

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return imageio.imread(path + file_name)

def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=384, hrg=384, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def modcrop(img, scale=4):
    """
        To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
    """
    # Check the image is grayscale
    if len(img.shape) == 3:
        h, w, _ = img.shape
        h = int(h / scale) * scale
        w = int(w / scale) * scale
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h = int(h / scale) * scale
        w = int(w / scale) * scale
        img = img[0:h, 0:w]
    return img

def resize(path, fileName, scale=4):
    img = cv2.imread(fileName,1)
    label_ = modcrop(img, scale)
    input_ = cv2.resize(label_, None, fx=1.0/scale, fy=1.0/scale,interpolation=cv2.INTER_CUBIC)  # Resize by scaling factor
    cv2.imwrite(fileName.split('.')[0]+'.png',input_)