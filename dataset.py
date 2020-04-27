import torch
import random
import numpy as np
from pre_processing import *
import torch.nn as nn
from random import randint
from PIL import Image, ImageSequence
import glob
from torch.utils.data.dataset import Dataset

BATCH_SIZE = 2
IN_SIZE = 1024
OUT_SIZE = 1024
TRAIN_VALID_RATIO = 0.8
# train_index = random.sample(range(0, 30), 24) 
# test_index = list(set([i for i in range(0, 30)]) - set(train_index))
# print(train_index)
# print(test_index)

def CREMIDataTrain(image_path, mask_path, in_size=IN_SIZE, out_size=OUT_SIZE):
    image_arr = Image.open(str(image_path))
    mask_arr = Image.open(str(mask_path))

    img_as_np = []
    orig_img_as_np = []
    for i, img_as_img in enumerate(ImageSequence.Iterator(image_arr)):
        if i not in [idx for idx in range(0,20)]:
            singleImage_as_np = np.asarray(img_as_img)
            img_as_np.append(singleImage_as_np)
            orig_img_as_np.append(singleImage_as_np)

    msk_as_np = []
    orig_msk_as_np = []
    for j, label_as_img in enumerate(ImageSequence.Iterator(mask_arr)):
        if j not in [idx for idx in range(0,20)]:
            singleLabel_as_np = np.asarray(label_as_img)
            msk_as_np.append(singleLabel_as_np)
            orig_msk_as_np.append(singleLabel_as_np)

    img_as_np, orig_img_as_np = np.stack(img_as_np, axis=0), np.stack(orig_img_as_np, axis=0)
    msk_as_np, orig_msk_as_np = np.stack(msk_as_np, axis=0), np.stack(orig_msk_as_np, axis=0)

    img_as_np, msk_as_np = flip(img_as_np, msk_as_np)

    # Noise Determine {0: Gaussian_noise, 1: uniform_noise
    # if randint(0, 1):
    #     gaus_sd, gaus_mean = randint(0, 20), 0
    #     img_as_np = add_gaussian_noise(img_as_np, gaus_mean, gaus_sd)
    # else:
    #     l_bound, u_bound = randint(-20, 0), randint(0, 20)
    #     img_as_np = add_uniform_noise(img_as_np, l_bound, u_bound)

    # change brightness
    pix_add = randint(-20, 20)
    img_as_np = change_brightness(img_as_np, pix_add)


    img_as_np, orig_img_as_np = normalization2(img_as_np.astype(float), max=1, min=0), normalization2(
        orig_img_as_np.astype(float), max=1, min=0)
    # print(msk_as_np[0])
    msk_as_np, orig_msk_as_np = msk_as_np / 255, orig_msk_as_np / 255
    
    img_as_tensor = torch.from_numpy(img_as_np).float()
    msk_as_tensor = torch.from_numpy(msk_as_np).long()
    orig_img_as_tensor, orig_msk_as_tensor = torch.from_numpy(orig_img_as_np).float(), torch.from_numpy(
        orig_msk_as_np).long()

    img_as_tensor = torch.cat((img_as_tensor, orig_img_as_tensor), 0)
    msk_as_tensor = torch.cat((msk_as_tensor, orig_msk_as_tensor), 0)

    return (img_as_tensor, msk_as_tensor)


class ComDataset(Dataset):
    def __init__(self, dataToSlice, SLICES_COLLECT):
        self.n_slices = len(SLICES_COLLECT)
        self.data = dataToSlice
        # for i, data in enumerate(dataToSlice):
        #     self.data.append(data)

    def __getitem__(self, index):
        if self.n_slices == 1:
            return self.data[0][index]
        elif self.n_slices == 2:
            return self.data[0][index], self.data[1][index]
        elif self.n_slices == 3:
            return self.data[0][index], self.data[1][index], self.data[2][index]

    def __len__(self):
        return len(self.data[0])


def CREMIDataVal(image_path, mask_path, in_size=IN_SIZE, out_size=OUT_SIZE):

    image_arr = Image.open(str(image_path))
    mask_arr = Image.open(str(mask_path))

    img_as_np = []
    for i, img_as_img in enumerate(ImageSequence.Iterator(image_arr)):
        if i in [idx for idx in range(0,20)]:
            singleImage_as_np = np.asarray(img_as_img)
            img_as_np.append(singleImage_as_np)

    msk_as_np = []
    for j, label_as_img in enumerate(ImageSequence.Iterator(mask_arr)):
        if j in [idx for idx in range(0,20)]:
            singleLabel_as_np = np.asarray(label_as_img)
            msk_as_np.append(singleLabel_as_np)

    img_as_np = np.stack(img_as_np, axis=0)
    msk_as_np = np.stack(msk_as_np, axis=0)

    # Normalize
    img_as_np = normalization2(img_as_np.astype(float), max=1, min=0)
    msk_as_np = msk_as_np / 255

    img_as_tensor = torch.from_numpy(img_as_np).float()
    msk_as_tensor = torch.from_numpy(msk_as_np).long()

    return (img_as_tensor, msk_as_tensor)


class CREMIDataPreTrained(Dataset):

    def __init__(self, image_path, in_size=IN_SIZE, out_size=OUT_SIZE):
        self.lh_images_array = glob.glob(str(image_path) + "/epoch_30/*lh.png")
        self.lh_images_array = sorted([x.replace('lh.png', '') for x in self.lh_images_array])

        self.binary_images_array = sorted(glob.glob(str(image_path) + "/epoch_30/*[0-9].png"))
        self.binary_images_array = sorted([x.replace('.png', '') for x in self.binary_images_array])

        self.gt_images_array = sorted(glob.glob(str(image_path) + "/epoch_30/*gt.png"))
        self.gt_images_array = sorted([x.replace('gt.png', '') for x in self.gt_images_array])

        self.orig_images_array = sorted(glob.glob(str(image_path) + "/epoch_30/*org.png"))
        self.orig_images_array = sorted([x.replace('org.png', '') for x in self.orig_images_array])

        self.in_size, self.out_size = in_size, out_size
        self.data_len = len(self.binary_images_array) 

    def __getitem__(self, index):

        single_lh_image = self.lh_images_array[index]
        single_binary_image = self.binary_images_array[index]
        single_gt_image = self.gt_images_array[index]
        single_origin_image = self.orig_images_array[index]

        lh_as_img = Image.open(single_lh_image + 'lh.png')
        binary_as_img = Image.open(single_binary_image + '.png')
        gt_as_img = Image.open(single_gt_image + 'gt.png')
        origin_as_img = Image.open(single_origin_image + 'org.png')

        lh_as_np = np.asarray(lh_as_img) / 255 # normalization?
        binary_as_np = np.asarray(binary_as_img) / 255
        gt_as_np = np.asarray(gt_as_img) / 255
        org_as_np = np.asarray(origin_as_img) / 255   

        lh_as_tensor = torch.from_numpy(lh_as_np).float()
        binary_as_tensor = torch.from_numpy(binary_as_np).long()
        gt_as_tensor = torch.from_numpy(gt_as_np).long()
        org_as_tensor = torch.from_numpy(org_as_np).float()

        return org_as_tensor, binary_as_tensor, gt_as_tensor

    def __len__(self):
        return self.data_len



if __name__ == "__main__":    
    x = random.sample(range(0, 4), 3)
    z = [0, 1, 2, 3]
    print(z[-2:])
    ss
    # print(set(z) - set(x))
    # ss
    y = np.array([22, 33,44,55])
    # print(y[x])
    train = CREMIDataPreTrained('history/UNET/result_images3')
    for i in range(26):
        train.__getitem__(i)
