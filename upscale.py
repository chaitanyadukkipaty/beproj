import os
import time
import random
import numpy as np
import scipy, multiprocessing
import tensorflow as tf
import tensorlayer as tl
from model import get_G, get_D, get_espcn
from config import config
from utils import *
import argparse
import glob
import cv2

checkpoint_dir = "models"
output_dir = "testOutput"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='multi', help='Upscale single or folder of images')
    parser.add_argument('--model', type=str, default='g110.h5', help='Model name')
    parser.add_argument('--data', type=str, default='Set14', help='Test dataset')
    parser.add_argument('--loc', type=str, default="resized", help='Location on data')
    args = parser.parse_args()

       

    if(args.model == 'g_srgan.npz'):
        G = get_G([1, None, None, 3])
        G.load_weights(os.path.join(checkpoint_dir, 'g_srgan.npz'))
    else:
        G = get_espcn([1, None, None, 3])
        G.load_weights(os.path.join(checkpoint_dir, args.model))
    G.eval()
    total_time = 0
    if(args.type == 'multi'):
        path = os.path.join(output_dir, args.data+'_'+args.model.split('.')[0])
        os.mkdir(path)
        data_HR = glob.glob(os.path.join('test',args.data+'/*'))
        c = len(args.data) + 6
        for i in data_HR:
            fileName =i[c::]
            outputName = fileName.split('.')[0]+'.png'
            #print(fileName,outputName)
            valid_lr_img = get_imgs_fn(fileName,'test/'+args.data+'/')  
            valid_lr_img = (valid_lr_img / 127.5) - 1  

            

            valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)
        
            if(len(valid_lr_img.shape)==3):
                valid_lr_img = valid_lr_img[np.newaxis,:,:,:]   
            else:
                valid_lr_img = np.stack((valid_lr_img,)*3, axis=-1)
                valid_lr_img = valid_lr_img[np.newaxis,:,:,:]
            start_time = time.time()
            out = G(valid_lr_img).numpy()
            print("--- %s seconds ---" % (time.time() - start_time))
            total_time += (time.time() - start_time) 

            print("[*] save images")
            tl.vis.save_image(out[0], os.path.join(path, outputName))   
        print("--- %s seconds ---" % (total_time))
        print("--- %s seconds ---" % (total_time/(len(data_HR)*1.0)))
        
    else:
        valid_lr_img = get_imgs_fn(args.data, 'data2017/')
        valid_lr_img = (valid_lr_img / 127.5) - 1 
        file_name = args.data.split('.')[0]+'.png'
        valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)
        valid_lr_img = valid_lr_img[np.newaxis,:,:,:]
        size = [valid_lr_img.shape[1], valid_lr_img.shape[2]]

        out = G(valid_lr_img).numpy()

        print("[*] save images")
        tl.vis.save_image(out[0], os.path.join('output', file_name))