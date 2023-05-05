from functools import partial
import numpy as np
from numpy import random
import time
from skimage import exposure
from skimage.transform import resize
import cv2
from model_2 import SalGAN_3, LOSS,SalGAN_4,SalGAN_2
import re, os, glob
import cv2
import torch
import scipy.misc
from torchvision import utils
from PIL import Image
from attention import Sal_based_Attention_module
from salgan import SalGAN_Generator


def normalize(x, method='standard', axis=None):
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res

def CC(saliency_map1, saliency_map2):
	map1 = np.array(saliency_map1, copy=False)
	map2 = np.array(saliency_map2, copy=False)
	if map1.shape != map2.shape:
		map1 = resize(map1, map2.shape, order=3, mode='constant')
	map1 = normalize(map1, method='standard')
	map2 = normalize(map2, method='standard')
	return np.corrcoef(map1.ravel(), map2.ravel())[0,1]



#Contrastive

model = SalGAN_4()

weight = torch.load('./models/model_encoder:_resnet_pool_decoder:_rand_lr:_0.0001_bz:_64_opt:_Adam/ckpt_epoch_56.pth')
#weight = torch.load('./trained_models/random_decoder/final_models/finetuned.pth')

model.load_state_dict(weight['model'], strict=True)


"""
#Salgan
model = SalGAN_Generator()
weight = torch.load('./salgan_3dconv_module.pt')

model.load_state_dict(weight, strict=False)

"""

"""
#Attention
model = Sal_based_Attention_module()

weight = torch.load('./attention.pt')
model.load_state_dict(weight['state_dict'], strict=False)

"""


model.cuda()


IMG_Path = "/home/yasser/Desktop/DATA360/images/"


image_list = os.listdir(IMG_Path)


if not os.path.isdir('./V'):
    os.makedirs('./V')

if not os.path.isdir('./saliency'):
    os.makedirs('./saliency')


print(image_list)
# i =80
for img in image_list:
    image_path = IMG_Path + img
    print(img)
    ori = cv2.imread(image_path)
    inpt = cv2.resize(ori,(320, 160))

    inpt = np.float32(inpt)
    #inpt-=[0.485, 0.456, 0.406]
    inpt = torch.cuda.FloatTensor(inpt)

    inpt = inpt.permute(2, 0, 1)
    
    inpt = torch.cuda.FloatTensor(inpt)

    with torch.no_grad():
        saliency_map = model(inpt.unsqueeze(0))

    Output = saliency_map
    Output = (Output.cpu()).detach().numpy()
    Output = Output.squeeze()
    Output = resize(Output, (1024,2048))
    np.save('./V/'+img[:-4]+'.npy',Output)
    cv2.imwrite('./saliency/'+img[:-4]+'.png',(Output-Output.min())*255/(Output.max()-Output.min()))
