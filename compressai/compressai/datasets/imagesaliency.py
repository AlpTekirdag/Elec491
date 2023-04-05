from pathlib import Path

from PIL import Image, ImageOps
import numpy as np
from torch.utils.data import Dataset

from compressai.registry import register_dataset

import warnings
warnings.filterwarnings('ignore')
import torch
from torchvision import transforms

@register_dataset("ImageFolderSaliency")
class ImageFolderSaliency(Dataset):
    """
        - rootdir/
            - train/
                - img000.png
            - test/
                - img000.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, patch_size=(256,256),split="train"):
        if(split == "train"):
            splitroot = Path(root) / "Salient2017"
            splitdir = Path(splitroot) / "Images"
            splitsal = Path(splitroot) / "Saliency"
        else:
            splitroot = Path(root) / "Eval"
            splitdir = Path(splitroot) / "Images"
            splitsal = Path(splitroot) / "Saliency"

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samplesImg = sorted(f for f in splitdir.iterdir() if f.is_file())
        self.samplesSal = sorted(f for f in splitsal.iterdir() if f.is_file())

        self.patch_size = patch_size
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img_ori = Image.open(self.samplesImg[index]).convert("RGB")
        img_ori = img_ori.resize((1024, 2048))
        sal = Image.open(self.samplesSal[index]).convert("L")
        img_ori_ =np.asarray(img_ori)
        h,w,c =img_ori_.shape
        sal_res = sal.resize((w, h))
        sal_res = np.asarray(sal_res)
        """sal_res = np.expand_dims(sal_res, axis=2)
        concat = np.concatenate((img_ori,sal_res),axis=2)
        img = Image.fromarray(np.uint8(concat))"""

        sal_res = torch.from_numpy(sal_res)
        sal_res = sal_res.unsqueeze(0)
        img_tensor = transforms.PILToTensor()(img_ori)
        if self.transform:
            crop_operation = self.transform.transforms[0]
            tensor_operation = self.transform.transforms[1]

            before_transformation = torch.cat((img_tensor, sal_res), dim=0)
            after_crop = crop_operation(before_transformation)

            sal_after_crop = after_crop[-1,:,:]
            sal_after_crop = sal_after_crop.unsqueeze(0)
            img_after_crop = after_crop[:3,:,:]
            img_after_crop = tensor_operation(transforms.ToPILImage()(img_after_crop))
            #sal_res = crop_operation(sal_res)
            #sal_res = sal_res.unsqueeze(0)
            #print(img_transformed.size(),sal_res.size()) 
            return torch.cat((img_after_crop, sal_after_crop), dim=0) 
        return []

    def __len__(self):
        return len(self.samplesImg)
