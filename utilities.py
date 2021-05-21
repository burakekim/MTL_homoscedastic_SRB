import glob
import os
import numpy as np
from matplotlib import pyplot as plt

palette = {0 : (255, 0, 0), # buildings -red 
           1 : (0, 0 , 0) # rest of the mask - black
}

invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def visualize(fig_name = None, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
        if fig_name is not None:
           plt.savefig(r'D:\burak\MULTI_TASK\TRIPLE_TASK\outputs\predictions\{}'.format(fig_name))     
    plt.show()
    
    
def get_id(FOLDER, test_ratio = 0.1, verbose = True):
    """
    Walk through dataset folder, extract all the IDs and divide them into train/val/test sets by using adjustable ratio parameter.
    Default: 
    0.7 Train
    0.2 Validation
    0.1 Test
    """

    image_dir = FOLDER + '\*'
    all_files = sorted(glob.glob(image_dir))
    ids = []

    for i in range(len(all_files)):
        all_things = os.path.basename(all_files[i]).split('.')
        all_things = all_things[0].split('_')
        first_id, second_id, patch_id = all_things[-4], all_things[-3], all_things[-1]
        ID = first_id + '_' + second_id + '_' + 'tile' + '_' + patch_id + '.tif'
        ids.append(ID)

    np.random.seed(0)
    test_ids = np.random.choice(ids, size=round(len(ids) * test_ratio), replace = False)
    validation_ids = np.random.choice(ids, size=round(len(ids) * (2 * test_ratio)), replace = False)
    
    train_val = np.setdiff1d(ids,test_ids)
    train_ids = np.setdiff1d(train_val, validation_ids)

    if verbose is not False: 
        print("len(train_ids): {}\nlen(validation_ids):{}\nlen(test_ids):{}\ntotal_#_of_ids:{}".format(len(train_ids),len(validation_ids),len(test_ids),len(ids)))
        return train_ids, validation_ids, test_ids
    else: 
        return train_ids, validation_ids, test_ids


