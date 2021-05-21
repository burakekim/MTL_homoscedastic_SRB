import albumentations as A
import cv2
import numpy as np
import skimage
import torch
from torch.utils.data import Dataset

from utilities import convert_from_color, get_id


class SemSeg_Custom(Dataset):
    """
    Outputs image, mask, boun_mask with the shape of (C,H,W) where C equals to N (number of classes). 
    On-the-fly cropping and boundary extraction operations. 
    """
    def __init__(
            self, 
            psp_dir,   
            masks_dir,
            mode='train',
            augmentation=None, 
            preprocessing=None,
            image_mask = False,
            inference = False):

        self.masks_dir = masks_dir
        self.psp_dir = psp_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.class_values = [0,1] # class values to encode 
        self.image_mask = image_mask 
        self.inference = inference
        
        train_ids, validation_ids, test_ids = get_id(psp_dir, verbose=False)        
     
        print("Len train_ids: {}, Len val_ids {}, Len test_ids {}".format(len(train_ids),len(validation_ids),len(test_ids)))
        masks_dir_ = masks_dir + '\{}'
        psp_dir_ = psp_dir + '\SN6_Train_AOI_11_Rotterdam_PS-RGB_{}'
        
        if mode == 'train':
            self.psp_list = [psp_dir_.format(id) for id in train_ids] 
            self.mask_list = [masks_dir_.format(id) for id in train_ids]
            
        elif mode == 'validation':
            self.psp_list = [psp_dir_.format(id) for id in validation_ids]       
            self.mask_list = [masks_dir_.format(id) for id in validation_ids]
           
        elif mode == 'test':
            self.psp_list = [psp_dir_.format(id) for id in test_ids]       
            self.mask_list = [masks_dir_.format(id) for id in test_ids]
           
    @staticmethod
    def get_boundary(label, kernel_size = (3,3)):
        tlabel = label.astype(np.uint8)
        temp = cv2.Canny(tlabel,0,1)
        tlabel = cv2.dilate(
                  temp,
                  cv2.getStructuringElement(
                  cv2.MORPH_CROSS,
                  kernel_size),
                  iterations = 2)
        tlabel = tlabel.astype(np.float32)
        tlabel /= 255.    
        return tlabel
    
    @staticmethod
    def _read_img(image_path):
        img = skimage.io.imread(image_path, plugin='tifffile')
        return img
        
    def __len__(self):
        return len(self.mask_list)
    
    def __getitem__(self, idx):     
           
        psp_filepath = self.psp_list[idx]
        mask_filepath = self.mask_list[idx]
        image = self._read_img(psp_filepath)
        mask = self._read_img(mask_filepath)  
            
        mask_raw = convert_from_color(mask)
        masks = [(mask_raw == v) for v in [0,1]]
        mask = np.stack(masks, axis=-1).astype('uint8') #(480, 480, 2))
            
        boun_mask = self.get_boundary(mask_raw) #(480, 480, 2))
        boun_mask = [(boun_mask == v) for v in [1,0]]
        boun_mask = np.stack(boun_mask, axis=-1).astype('uint8') #(480, 480, 2))  
            
        if self.augmentation:
            transformed = A.Compose(self.augmentation, p=1)(image=image, masks=[mask, boun_mask])
            
            image, mask, boun_mask = transformed['image'], transformed['masks'][0], transformed['masks'][1]

        if self.preprocessing:
            preprocessed  = self.preprocessing(image=image, mask=mask, boundary_mask = boun_mask)
            image, mask, boun_mask = preprocessed['image'], preprocessed['mask'], preprocessed['boundary_mask']

        image = image[...] / 255.0
        
        image = np.asarray(image).transpose(2,0,1)
        mask = np.asarray(mask).transpose(2,0,1)
        boun_mask = np.asarray(boun_mask).transpose(2,0,1)
        
        image = torch.as_tensor(image, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.float32)
        boun_mask = torch.as_tensor(boun_mask, dtype=torch.float32)
        
        if self.inference == True:
            return image,mask,boun_mask,self.psp_list        
        if self.image_mask == True :
            return image, mask
        if self.image_mask == False:
            return image, mask, boun_mask

def get_training_augmentation(crop_size):
    train_transform = [
        
        A.OneOf([A.RandomCrop(crop_size[0], crop_size[1], p=1.0)
                                       ], p=1.0),
    
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.7,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.7,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.7,
        ),
        

    ]

    return A.Compose(train_transform)

def get_val_augmentation(crop_size):
    val_transform = [
        
        A.OneOf([A.RandomCrop(crop_size[0], crop_size[1], p=1.0)
                                       ], p=1.0),]

    return A.Compose(val_transform)

def get_test_augmentation(crop_size): # ensure determinism. replace random crop with center crop. 
    val_transform = [
        
        A.OneOf([A.CenterCrop(crop_size[0], crop_size[1], p=1.0)
                                       ], p=1.0),]

    return A.Compose(val_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2,0,1).astype('float32')

#preprocessing_fn = get_preprocessing_fn(params['encoder'], params['encoder_weights'])

def get_preprocessing(preprocessing_fn):
   
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask= to_tensor),
    ]
    return A.Compose(_transform)
