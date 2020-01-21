import cv2
import glob
import os
from skimage import measure
import argparse

save_dir = "samples"

def modcrop(img, scale=4):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--og', type=str, default='lenna.png', help='Original imgae name')
    parser.add_argument('--gen', type=str, default='gan110.png', help='Generated image name')
    args = parser.parse_args()

    data_HR = glob.glob(os.path.join('data2017',args.og))
    print(data_HR)
    data_LR = glob.glob(os.path.join(save_dir,args.gen))
    print(data_LR)
    hr = modcrop(cv2.imread(data_HR[0]))
    lr = cv2.imread(data_LR[0])
    lr = modcrop(cv2.resize(lr,None,fx = 1.0/4 ,fy = 1.0/4, interpolation = cv2.INTER_CUBIC))
    print(measure.compare_psnr(lr, hr))
    print(measure.compare_ssim(lr,hr,multichannel=True))
