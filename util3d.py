import torch
import os
import random
import numpy as np
from pre_processing import *
import torch.nn as nn
from random import randint
from PIL import Image, ImageSequence
import glob
from torch.utils.data.dataset import Dataset
def CREMIDataVal(image_path, mask_path):

    image_arr = Image.open(str(image_path))
    mask_arr = Image.open(str(mask_path))

    img_as_np = []
    for i, img_as_img in enumerate(ImageSequence.Iterator(image_arr)):
        if i in [idx for idx in range(0,20)]:
            singleImage_as_np = np.asarray(img_as_img)
            for r in range(0,1024, 256):
                for c in range(0,1024, 256):
                    img_as_np.append(singleImage_as_np[r:r+256,c:c+256])

    msk_as_np = []
    for j, label_as_img in enumerate(ImageSequence.Iterator(mask_arr)):
        if j in [idx for idx in range(0,20)]:
            singleLabel_as_np = np.asarray(label_as_img)
            for r in range(0,1024, 256):
                for c in range(0,1024, 256):
                    msk_as_np.append(singleLabel_as_np[r:r+256,c:c+256])

    img_as_np = np.stack(img_as_np, axis=0)
    msk_as_np = np.stack(msk_as_np, axis=0)

    # Normalize
    img_as_np = normalization2(img_as_np.astype(float), max=1, min=0)
    msk_as_np = msk_as_np / 255

    img_as_tensor = torch.from_numpy(img_as_np).float()
    msk_as_tensor = torch.from_numpy(msk_as_np).long()

    return (img_as_tensor, msk_as_tensor)

def CREMIDataTrain(image_path, mask_path):
    image_arr = Image.open(str(image_path))
    mask_arr = Image.open(str(mask_path))

    img_as_np = []
    orig_img_as_np = []
    for i, img_as_img in enumerate(ImageSequence.Iterator(image_arr)):
        if i not in [idx for idx in range(0,20)]:
            singleImage_as_np = np.asarray(img_as_img)
            for r in range(0,1024, 256):
                for c in range(0,1024, 256):
                    img_as_np.append(singleImage_as_np[r:r+256,c:c+256])
                    orig_img_as_np.append(singleImage_as_np[r:r+256,c:c+256])

    msk_as_np = []
    orig_msk_as_np = []
    for j, label_as_img in enumerate(ImageSequence.Iterator(mask_arr)):
        if j not in [idx for idx in range(0,20)]:
            singleLabel_as_np = np.asarray(label_as_img)
            for r in range(0,1024, 256):
                for c in range(0,1024, 256):
                    msk_as_np.append(singleLabel_as_np[r:r+256,c:c+256])
                    orig_msk_as_np.append(singleLabel_as_np[r:r+256,c:c+256])

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


def prepareDataForLoader(data,slice):
    slice = slice[0]
    train, label = data
    print(train.shape, label.shape)

    paddingSize = int(slice / 2)
    paddings = torch.zeros(paddingSize, train.shape[2], train.shape[2])
    train = torch.cat((paddings, train, paddings))
    label = torch.cat((paddings.long(), label, paddings.long()))

    new_train = []
    new_label = []
    for i in range(0, train.shape[0] - slice + 1):
        new_train.append(train[i:i + slice])
        new_label.append(label[i:i + slice])

    new_train = torch.stack(new_train).unsqueeze(1)
    new_label = torch.stack(new_label)
    print(new_train.shape, new_label.shape)

    train_data = []
    for i in range(new_label.shape[0]):
        train_data.append([new_train[i], new_label[i]]) # [[1, 3, 1250, 1250], [1, 3, 1250, 1250]]

    return [train_data]

class ComDataset(Dataset):
    def __init__(self, dataToSlice):
        self.data = dataToSlice

    def __getitem__(self, index):
        return self.data[0][index]

    def __len__(self):
        return len(self.data[0])

def get_3d_dataset(dataset_path, dataset_cache, SLICES_COLLECT):
    train_path = dataset_path + '/train-volume.tif'
    val_path = dataset_path + '/train-labels_thin.tif'
    
    if dataset_cache and os.path.isfile(dataset_cache):
        print("Load enhanced dataset before DataLoader from cache at %s", dataset_cache)
        saved_data = torch.load(dataset_cache)
        trainDataset = prepareDataForLoader(saved_data[0], SLICES_COLLECT)
        validDataset = prepareDataForLoader(saved_data[1], SLICES_COLLECT)
    else:
        print("Start Prepare enhanced dataset before DataLoader %s", dataset_path)
        trainData = CREMIDataTrain(train_path, val_path)
        validData = CREMIDataVal(train_path, val_path)
        saved_data = [trainData,validData]
        torch.save(saved_data,dataset_cache)
        trainDataset = prepareDataForLoader(trainData, SLICES_COLLECT)
        validDataset = prepareDataForLoader(validData, SLICES_COLLECT)
        print("list train and valid as the dataset")
    # print(len(trainDataset), len(trainDataset[0]), trainDataset[0][0][0].shape)
    
    dataset = [ComDataset(trainDataset), ComDataset(validDataset)]
    # """
    # x = dataset[0].__getitem__(0)
    # print(x.shape)  # (2, (1, 1250, 1250), (1250, 1250)) [2 torch.Size([1, 3, 1250, 1250]) torch.Size([3, 1250, 1250])]
    # """

    return dataset


