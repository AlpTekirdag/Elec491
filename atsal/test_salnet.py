import torch
import time
import os
import cv2
import numpy as np
from PIL import Image

from refinenet import RefineNet

from torch.utils.data import DataLoader
from train_loader import Train_dataset

from losses.saloss import Saloss
import itertools

from model import Sal_based_Attention_module, SalEMA
import projection_methods

torch.autograd.set_detect_anomaly(True)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def load_model(pt_model, new_model):
    temp = torch.load('./weight/'+pt_model+'.pt')['state_dict']
    new_model.load_state_dict(temp)
    return new_model

if __name__ == '__main__':

    # print train start time
    print('start time -->', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # define super parameters
    BACT_SIZE = 1

    root = "datalar/"
    result_dst = "result/final"
    os.makedirs(result_dst, exist_ok=True)

    # load datset
    test_dataset = Train_dataset(root=root,split="test")

    # data loader
    test_loader = DataLoader(dataset=test_dataset, batch_size=BACT_SIZE, shuffle=True, num_workers=4)
    
    ## ATSAL
    assert torch.cuda.is_available(), \
        "CUDA is not available in your machine"
    # Create networks
    att_model = Sal_based_Attention_module()
    salema_copie = SalEMA()
    refine_model = RefineNet()

    # load weight
    att_model = load_model("attention", att_model).cuda()
    Poles  = load_model("poles", salema_copie).cuda()
    Equator  = load_model("equator", salema_copie).cuda()
    temp = torch.load('./weight/refine.pth')['refinemodel_state_dict']
    refine_model.load_state_dict(temp)
    refine_model = refine_model.cuda()
    #refine_model = load_model("refine",refine_model).cuda()

    refine_lossfc = Saloss()
    # nss_lossfc = NSS()
    # cc_lossfc = CC()
    # kld_lossfc = torch.nn.KLDivLoss()

    '''
                ========================= Test ================================
    '''

    refine_model.eval()
    # Validation

    val_refine_loss = 0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            img, sal = batch
            img = img.cuda()
            sal = sal.cuda()
            out_state = {'F':None, 'R':None, 'B':None, 'L':None, 'U':None, 'D':None}
            state = None    

            refine_loss = 0

            start_time = time.time()
            img = img.cpu().numpy()
            img = img.astype(np.float32)
            norm_RGB = [103.939, 116.779, 123.68]
            norm_RGB = np.array(norm_RGB)
            norm_RGB = norm_RGB[:,None,None]
            img -= norm_RGB
            img = torch.FloatTensor(img)
            img = img.cuda() # Shape => B, 3, 640, 320
            sal = sal.permute(0,1,3,2) # Shape =>B, 1, 640, 320 => B, 1, 320, 640

            first_stream = ((att_model(img))).permute(0,1,3,2)

            #img = img.squeeze()
            img = img.permute(0,3,2,1)
            img = img.cpu().numpy()
            B,w,h,c = img.shape

            # Expert module output
            expert_res =np.zeros((B,1,w,h))

            for j, im in enumerate(img):
                if len(im.shape) == 2 :
                    im = im[..., None]            
                out = projection_methods.e2c(im, face_w=256 , mode='bilinear', cube_format='dict')
                out_predict ={}
                for face_key in out:
                    cmp_face = out[face_key].astype(np.float32)
                    state = out_state[face_key]
                    if len(cmp_face.shape) == 2 :
                        cmp_face = cmp_face[..., None]
                    cmp_face -= [103.939, 116.779, 123.68]
                    cmp_face = torch.cuda.FloatTensor(cmp_face)
                    cmp_face = cmp_face.permute(2,0,1)
                    cmp_face = cmp_face.unsqueeze(0)
                    if face_key == 'U' or face_key == 'D':
                        state, out_face = Poles.forward(input_ = cmp_face, prev_state = state)   
                    else :
                        state, out_face = Equator.forward(input_ = cmp_face, prev_state = state)
                    state = repackage_hidden(state)
                    out_face = out_face.squeeze()
                    out_face = out_face.cpu().detach().numpy()
                    out_face = out_face*255/out_face.max()
                    out_face = cv2.resize(out_face,(256,256))
                    if len(out_face.shape) == 2 :
                        out_face = out_face[..., None]
                    out_predict[face_key] = np.array(out_face.astype(np.uint8))
                    out_state[face_key]   = state 
                
                second_stream = (projection_methods.c2e(out_predict, h=320, w=640, mode='bilinear',cube_format='dict')).reshape(320,640)
                second_stream= second_stream/255
                second_stream = second_stream[None,:,:]
                expert_res[j,:,:,:] = second_stream


            second_stream = torch.from_numpy(expert_res).cuda()            
            second_stream = second_stream.type(torch.float32)

            saliency_map = refine_model(first_stream, second_stream)
            saliency_clone = saliency_map
            saliency_map = saliency_map.cpu().numpy()
            saliency_map = saliency_map.squeeze()
            cv2.imwrite(os.path.join(result_dst,str(count).zfill(4)+'.png'),(255*saliency_map/saliency_map.max()).astype(np.uint8))
            val_refine_loss += refine_lossfc(saliency_clone, sal).detach().item()
            count+=1

        # test Loss
        print('Number of Test:{}, test_loss:{:.8f}'.format(count, val_refine_loss / count))

    # print train end time
    print('end time -->', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))



