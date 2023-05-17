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
    BACT_SIZE = 4
    #LR = 1.30208333333e-07
    LR = 1.3e-06
    EPOCH = 50  
    root = "datalar/"

    # load datset
    train_dataset = Train_dataset(root=root)

    # Train & Val
    train_db, val_db = torch.utils.data.random_split(train_dataset, [0.8, 0.2])

    # batch input
    train_loader = DataLoader(dataset=train_db, batch_size=BACT_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_db, batch_size=BACT_SIZE, shuffle=True, num_workers=4)
    
    ## ATSAL
    assert torch.cuda.is_available(), \
        "CUDA is not available in your machine"
    # Create networks
    att_model = Sal_based_Attention_module()
    salema_copie = SalEMA()


    # load weight
    att_model = load_model("attention", att_model).cuda()
    Poles  = load_model("poles", salema_copie).cuda()
    Equator  = load_model("equator", salema_copie).cuda()

    refine_model = RefineNet()
    refine_model = refine_model.cuda()

    refine_lossfc = Saloss()
    # nss_lossfc = NSS()
    # cc_lossfc = CC()
    # kld_lossfc = torch.nn.KLDivLoss()

    refine_optim = torch.optim.Adam(refine_model.parameters(), lr=LR, weight_decay=5e-07)

    path = 'weight/basenet_saloss_refinenet_saloss_9+5_lr_' + str(LR) + '_' + str(EPOCH)
    model_path = path

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(model_path + '/refine', exist_ok=True)

    # visualize loss value in the browser by tensorboard
    iter = 0
    
    '''
    ========================= train ================================
    '''
    
    for epoch in range(EPOCH):
        
        refine_model.train()

        for i, batch in enumerate(train_loader):
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

            img = img.squeeze()
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


            ## REfine net test
            saliency_map = refine_model(first_stream, second_stream)
            saliency_clone = saliency_map

            refine_loss += refine_lossfc(saliency_clone, sal)

            refine_optim.zero_grad()
            refine_loss.backward()
            refine_optim.step()
            
            # Save model
            if i % 1000 == 0:

                
                ################################  refine info ###################################
                print('REFINE Train EPOCH:{}, iteration:{}, train_refine_loss:{:.8f}'.format(epoch, iter, refine_loss))
                refine_save_path = model_path + '/' + 'refine' + '/' + str(epoch) + '_' + 'iter' + str(iter) + '.pt'
                # Save model
                refine_state = {'refine_model': refine_model,
                                'refinemodel_state_dict': refine_model.state_dict(),
                                'refine_optim_state_dict': refine_optim.state_dict(),
                                'epoch': epoch,
                                'refine_loss': refine_loss.detach().item()}
                torch.save(refine_state, refine_save_path)

                iter = iter + 1
                #"""

        '''
                    ========================= val ================================
        '''

        refine_model.eval()
        # Validation

        val_refine_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
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

                img = img.squeeze()
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

                val_refine_loss += refine_lossfc(saliency_clone, sal).detach().item()

            # Val Loss
            print('EPOCH:{}, val_refine_loss:{:.8f}'.format(epoch, val_refine_loss / len(val_loader)))

    # print train end time
    print('end time -->', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))



