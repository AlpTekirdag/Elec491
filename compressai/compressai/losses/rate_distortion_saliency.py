import math
import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim
from compressai.registry import register_criterion
import numpy as np


@register_criterion("RateDistortionLossSaliency")
class RateDistortionLossSaliency(nn.Module):

    def __init__(self, lmbda=0.01, metric="mse", return_type="all"):
        super().__init__()
        if metric == "mse":
            self.metric = nn.MSELoss(reduction ="none")
        elif metric == "ms-ssim":
            self.metric = ms_ssim
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.return_type = return_type

    def forward(self, output, target):
        N, C, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        #print(target[:,1,:,:].min(),target[:,1,:,:].max())
        #print(target[:,-1,:,:].min(),target[:,-1,:,:].max())
        #mask = target[:,-1,:,:]/255.
        mask = target[:,-1,:,:]
        #print(mask.min(),mask.max())
        mask = nn.Sigmoid()(mask) #for latent masking 
        #print(mask.min(),mask.max())
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, 3, 1, 1)
        
        target_ = target[:,:3,:,:] ## ALP

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.metric == ms_ssim:
            out["ms_ssim_loss"] = self.metric(output["x_hat"], target_, data_range=1)
            distortion = 1 - out["ms_ssim_loss"]
        else:
            mse_loss_initial = self.metric(output["x_hat"], target_)
            mse_loss_masked = (mse_loss_initial * mask.float()).sum()
            non_zero_elements = mask.sum()
            mse_loss_masked = mse_loss_masked / non_zero_elements
            out["mse_loss"] = mse_loss_masked
            distortion = 255**2 * out["mse_loss"]

        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
