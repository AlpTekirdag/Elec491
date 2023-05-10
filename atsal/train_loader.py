import torch
import os
from pathlib import Path
import datetime
import numpy as np
import cv2
from PIL import Image, ImageOps
from torchvision import transforms

from torch.utils import data
from torchvision import utils

# The DataLoader for our specific datataset with extracted frames
class Train_dataset(data.Dataset):

    def __init__(self, root, split="train"):
        
        if(split == "train"):
            splitroot = Path(root) / "Train"
            splitdir = Path(splitroot) / "train"
            splitsal = Path(splitroot) / "train_saliency"
        else:
            splitroot = Path(root) / "Test"
            splitdir = Path(splitroot) / "test"
            splitsal = Path(splitroot) / "test_saliency"

        # Gives accurate human readable time, rounded down not to include too many decimals
        self.samplesImg= sorted(f for f in splitdir.iterdir() if f.is_file())
        self.samplesSal  = sorted(f for f in splitsal.iterdir() if f.is_file())

        print(len(self.samplesImg))

        print('data loaded')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samplesImg)
    
    def __getitem__(self, index):

        'Generates one sample of data'
        width = 320
        height = 640
        img_ori = Image.open(self.samplesImg[index]).convert("RGB")
        img_ori = img_ori.resize((width, height))

        sal = Image.open(self.samplesSal[index]).convert("L")
        sal_res = sal.resize((width, height))

        sal_res = transforms.PILToTensor()(sal_res)
        img_tensor = transforms.PILToTensor()(img_ori)
        
        packed = (img_tensor,sal_res)

        return packed