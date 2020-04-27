import logging
import torch
import os
from dataset import *

logger = logging.getLogger(__file__)


def toSlicesGroupDataset(train, label, slice):
    print('SLICES: ', slice)
    if slice == 1:
        return [[train[i].unsqueeze(0), label[i]] for i in range(label.shape[0])]

    paddingSize = int(slice / 2)
    paddings = torch.zeros(paddingSize, train.shape[2], train.shape[2])
    train = torch.cat((paddings, train, paddings))

    new_train = []
    for i in range(0, train.shape[0] - slice + 1):
        new_train.append(train[i:i + slice])

    new_train = torch.stack(new_train)

    train_data = []
    for i in range(label.shape[0]):
        train_data.append([new_train[i], label[i]])

    return train_data


def prepareDataForLoader(data, SLICES_COLLECT):
    train, label = data
    # print(len(label), label[0])
    dataToSlice = []

    for i, SLICE in enumerate(SLICES_COLLECT):
        dataToSlice.append(toSlicesGroupDataset(train, label, SLICE))

    return dataToSlice
    
# def get_dataset(dataset_path, dataset_cache, SLICES_COLLECT):
#     train_path = dataset_path + '/train-volume.tif'
#     val_path = dataset_path + '/train-labels_thin.tif'
#     for SLICE in SLICES_COLLECT:
#         extension = str(SLICE)
#         dataset_cache = dataset_cache + '_' + str(extension)

#     if dataset_cache and os.path.isfile(dataset_cache):
#         print("Load enhanced dataset before DataLoader from cache at %s", dataset_cache)
#         dataset = torch.load(dataset_cache)
#     else:
#         print("Start Prepare enhanced dataset before DataLoader %s", dataset_path)

#         trainData = CREMIDataTrain(train_path, val_path)
#         validData = CREMIDataVal(train_path, val_path)

#         trainDataset = prepareDataForLoader(trainData, SLICES_COLLECT)
#         validDataset = prepareDataForLoader(validData, SLICES_COLLECT)
#         print("list train and valid as the dataset")
#         dataset = [trainDataset, validDataset]
#         torch.save(dataset, dataset_cache)
#     # x1 = ComDataset(dataset[0])
#     # print(len(x1),len(x1[0]), x1[0][1].shape)
#     dataset = [ComDataset(dataset[0], SLICES_COLLECT), ComDataset(dataset[1], SLICES_COLLECT)]
#     # """
#     # x1, x2, x3 = dataset.__getitem__(0)
#     # print(len(x1), x1[0].shape, x1[1].shape)  # (2, (1, 1250, 1250), (1250, 1250))
#     # print(len(x2), x2[0].shape, x2[1].shape)  # (2, (3, 1250, 1250), (1250, 1250))
#     # print(len(x3), x3[0].shape, x3[1].shape)  # (2, (5, 1250, 1250), (1250, 1250))
#     # """
#     return dataset

def get_dataset(dataset_path, dataset_cache, SLICES_COLLECT):
    train_path = dataset_path + '/train-volume.tif'
    val_path = dataset_path + '/train-labels.tif'
    for SLICE in SLICES_COLLECT:
        extension = str(SLICE)
        dataset_cache = dataset_cache #+ '_' + str(extension)

    if dataset_cache and os.path.isfile(dataset_cache):
        print("Load enhanced dataset before DataLoader from cache at %s", dataset_cache)
        saved_data = torch.load(dataset_cache)

        trainDataset = prepareDataForLoader(saved_data[0], SLICES_COLLECT)
        validDataset = prepareDataForLoader(saved_data[1], SLICES_COLLECT)
        dataset = [trainDataset, validDataset]
    else:
        print("Start Prepare enhanced dataset before DataLoader %s", dataset_path)

        trainData = CREMIDataTrain(train_path, val_path)
        validData = CREMIDataVal(train_path, val_path)
        saved_data = [trainData,validData]
        torch.save(saved_data,dataset_cache)

        trainDataset = prepareDataForLoader(trainData, SLICES_COLLECT)
        validDataset = prepareDataForLoader(validData, SLICES_COLLECT)
        print("list train and valid as the dataset")
        dataset = [trainDataset, validDataset]
    print(len(trainDataset), len(trainDataset[0]), trainDataset[0][0][0].shape)
    # x1 = ComDataset(dataset[0])
    # print(len(x1),len(x1[0]), x1[0][1].shape)
    # ss
    dataset = [ComDataset(dataset[0], SLICES_COLLECT), ComDataset(dataset[1], SLICES_COLLECT)]
    return dataset


if __name__ == "__main__":
    # A full forward pass
    dataset_cache = 'dataset_cache'
    SLICES_COLLECT = [3]
    trainDataset, validDataset = get_dataset('train_allen', dataset_cache, SLICES_COLLECT)

    x1 = trainDataset.__getitem__(0)
    print(x1)
    # print(x2)
    print(len(x1), x1[0].shape, x1[1].shape)  # (2, (1, 1250, 1250), (1250, 1250))
    # print(len(x2), x2[0].shape, x2[1].shape)  # (2, (3, 1250, 1250), (1250, 1250))
    # print(len(x3), x3[0].shape, x3[1].shape)  # (2, (5, 1250, 1250), (1250, 1250))
