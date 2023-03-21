from pathlib import Path

from PIL import Image, ImageOps
import numpy as np
from torch.utils.data import Dataset

from compressai.registry import register_dataset


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
            splitroot = Path(root) / "Train"
            splitdir = Path(splitroot) / "train"
            splitsal = Path(splitroot) / "train_saliency"
        else:
            splitroot = Path(root) / "Test"
            splitdir = Path(splitroot) / "test"
            splitsal = Path(splitroot) / "test_saliency"

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
        img_ori =np.asarray(img_ori)
        h,w,c =img_ori.shape
        sal_res = sal.resize((w, h))
        sal_res = np.asarray(sal_res)
        sal_res = np.expand_dims(sal_res, axis=2)
        concat = np.concatenate((img_ori,sal_res),axis=2)
        img = Image.fromarray(np.uint8(concat))

        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samplesImg)
