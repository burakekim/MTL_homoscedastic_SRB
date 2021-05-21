
import numpy as np
import os
import torch

def get_params():

    params = {}

    params['experiment_log'] = "MTL_ser_r101u_0" # experiment ID.     
    params['main_dir'] = r"C:\Users\burak\Desktop"
    params['log_path'] = os.path.join(params['main_dir'], 'mtl', 'logs') # path to save logs.

    params['main_data_dir'] = r"D:\Veri-Setleri\SN6\SN6_buildings_AOI_11_Rotterdam_train\train\AOI_11_Rotterdam"
    params['masks_dir'] = os.path.join(params['main_data_dir'], 'MASKS_binary') # dataset path. 
    params['psp_dir'] = os.path.join(params['main_data_dir'], 'PS-RGB') # dataset path. 

    params['crop_size'] = (480,480) 
    params['encoder'] = 'resnet101' # backbone architecture, encoder. 
    params['encoder_weights'] = 'imagenet' # pre-trained weights. 
    params['classes'] = np.arange(0,2,1) # used for encoding-decoding the mask.['building','not building']
    params['activation'] = 'sigmoid' # activation function.
    params['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU or CPU.
    params['batch_size'] = 4 # batch size. 
    params['lr'] = 0.0001 # learning rate.
    params['n_epoch'] = 50 # number of epochs.
    params['n_workers'] = 0 #number of workers for data loader, multi-process scheme.
    params['seed'] = 12

    return params

