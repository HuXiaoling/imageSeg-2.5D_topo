from model import U_Net
from model3D import UNet
from dataset import *
from modules import *
from utils import *
from save_history import *
import torch
import os
import numpy as np
import torch.nn as nn
from util3d import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# np.set_printoptions(threshold=sys.maxsize)

def multiModel(SLICES_COLLECT):
    
    models = []
    for SLICE in SLICES_COLLECT:
        models.append(U_Net(in_channels=SLICE, out_channels = 32)) #out_channels = 32

    return models

def pre_train():

    if device.type == "cuda":
        print("GPU: ", torch.cuda.device_count())
    for i in range(len(models)):
        models[i] = torch.nn.DataParallel(models[i], device_ids=list(
            range(torch.cuda.device_count()))).cuda()

    loss_fun = nn.CrossEntropyLoss()
    optimizers = []
    for i in range(len(models)):
        optimizers.append(torch.optim.RMSprop(models[i].parameters(), lr=LR)) #
    # Train
    print("Initializing Training!")
    for i in range(0, pretrain_epoch):
        train_multi_models(models,
                           train_load,
                           loss_fun,
                           optimizers,
                           device,
                           SLICES_COLLECT)

        # just for print loss
        train_acc, train_loss = get_loss_train(models,
                                               train_load,
                                               loss_fun,
                                               device,
                                               SLICES_COLLECT)

        print('Epoch', str(i + 1), 'Train loss:', train_loss, "Train acc", train_acc)

        # Validation every 5 epoch
        if (i + 1) % 5 == 0:
            val_acc, val_loss = validate_model(
                models,
                val_load,
                loss_fun,
                i + 1,
                True,
                image_save_path,
                device,
                SLICES_COLLECT)
            print('Val loss:', val_loss, "val acc:", val_acc)

            values = [i + 1, train_loss, train_acc, val_loss, val_acc]
            export_history(header, values, save_dir, save_file_name)

            if (i + 1) % 10 == 0:  # save model every 10 epoch
                save_models(models, model_save_dir, i + 1, SLICES_COLLECT)

def topo_train(models):
    if device.type == "cuda":
        print("GPU: ", torch.cuda.device_count())
    for i in range(len(models)):
        # models[i].to(device)
        models[i] = torch.nn.DataParallel(models[i], device_ids=list(
            range(torch.cuda.device_count()))).cuda()

    loss_fun = nn.CrossEntropyLoss()
    optimizers = []
    for i in range(len(models)):
        optimizers.append(torch.optim.RMSprop(models[i].parameters(), lr=LR))
    # Train
    print("Initializing Topo Training!")
    for i in range(pretrain_epoch, topo_epoch):
        train_topo_multi_models(models,
                           train_load,
                           loss_fun,
                           optimizers,
                           device,
                           SLICES_COLLECT,
                           i,
                           save_dir)
        # just for print loss
        train_acc, train_loss = get_loss_train(models,
                                               train_load,
                                               loss_fun,
                                               device,
                                               SLICES_COLLECT)

        print('Epoch', str(i + 1), 'Train loss:', train_loss, "Train acc", train_acc)

        # Validation every 5 epoch
        if (i + 1) % 1 == 0:
            val_acc, val_loss = validate_model(
                models,
                val_load,
                loss_fun,
                i + 1,
                True,
                image_save_path,
                device,
                SLICES_COLLECT)
            print('Val loss:', val_loss, "val acc:", val_acc)

            values = [i + 1, train_loss, train_acc, val_loss, val_acc]
            export_history(header, values, save_dir, save_file_name)

            if (i + 1) % 1 == 0:  # save model every 10 epoch
                save_models(models, model_save_dir, i + 1, SLICES_COLLECT)

def train():
    if device.type == "cuda":
        print("GPU: ", torch.cuda.device_count())
    for i in range(len(models)):
        models[i] = torch.nn.DataParallel(models[i], device_ids=list(
            range(torch.cuda.device_count()))).cuda()

    loss_fun = nn.CrossEntropyLoss()
    # loss_fun = nn.MSELoss()
    optimizers = []
    for i in range(len(models)):
        optimizers.append(torch.optim.RMSprop(models[i].parameters(), lr=LR))
    # Train
    print("Initializing Training!")
    for i in range(0, topo_epoch):
        if (i == pretrain_epoch):
            print("Initializing Topo Training!")

        train_topo_multi_models(models,
                           train_load,
                           loss_fun,
                           optimizers,
                           device,
                           SLICES_COLLECT,
                           i,
                           pretrain_epoch,
                           save_dir)

        # just for print loss
        train_acc, train_loss = get_loss_train(models,
                                               train_load,
                                               loss_fun,
                                               device,
                                               SLICES_COLLECT)

        print('Epoch', str(i + 1), 'Train loss:', train_loss, "Train acc", train_acc)

        # Validation every 5 epoch
        every = 5
        every_model = 10
        if i >= pretrain_epoch: 
            every = 1
            every_model = 1
        if (i + 1) % every == 0:
            val_acc, val_loss = validate_model(
                models,
                val_load,
                loss_fun,
                i + 1,
                True,
                image_save_path,
                device,
                SLICES_COLLECT)
            print('Val loss:', val_loss, "val acc:", val_acc)

            values = [i + 1, train_loss, train_acc, val_loss, val_acc]
            export_history(header, values, save_dir, save_file_name)

            if (i + 1) % every_model == 0:  # save model every 10 epoch
                save_models(models, model_save_dir, i + 1, SLICES_COLLECT)



def train3D():
    model = UNet(in_dim=1, out_dim=2, num_filters=32)
    if device.type == "cuda":
        print("GPU: ", torch.cuda.device_count())
    model = torch.nn.DataParallel(model, device_ids=list(
        range(torch.cuda.device_count()))).cuda()

    loss_fun = nn.CrossEntropyLoss()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR) #
    print("Initializing Training!")
    for i in range(0, pretrain_epoch):
        softmax = nn.Softmax2d()
        model.train()
        loss = 0
        for batch, (images, masks) in enumerate(train_load): 
            # print('model input shape: ', images.shape, masks.shape)
            # model input shape:  torch.Size([2, 1, 3, 1250, 1250]) torch.Size([2, 3, 1250, 1250])
            predict_maps = model(images.to(device))
            B,C,D,H,W=predict_maps.size()
            predict_maps=predict_maps.view(B,C,D, -1)
            masks = masks.view(B, D,-1)
            # print(predict_maps.shape) #[4, 2, 3, 625, 625]  #retain_graph=True
            loss = loss_fun(predict_maps, masks.to(device))
            # print(loss) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        model.eval()
        total_acc = 0
        total_loss = 0
        for batch, (images, masks) in enumerate(train_load):
            with torch.no_grad():
                predict_maps = model(images.to(device))
                B,C,D,H,W=predict_maps.size()
                predict_maps=predict_maps.view(B,C,D, -1)
                masks = masks.view(B, D,-1)
                loss = loss_fun(predict_maps, masks.to(device))
                pred_class = torch.argmax(predict_maps, dim=1).float()
                acc = accuracy_check_for_batch(masks.cpu(), pred_class.cpu(), masks.size()[0])
                total_acc += acc
                total_loss += loss.cpu().item()

        train_acc, train_loss = total_acc / (batch + 1), total_loss / (batch + 1)

        print('Epoch', str(i + 1), 'Train loss:', train_loss, "Train acc", train_acc)
        if (i + 1) % 5 == 0:
            total_val_loss = 0
            total_val_acc = 0
            softmax = nn.Softmax2d()
            for batch, (images, masks) in enumerate(val_load):
                with torch.no_grad():  
                    predict_maps = model(images.to(device))
                    B,C,D,H,W=predict_maps.size()
                    predict_maps=predict_maps.view(B,C,D, -1)
                    likelihoodMaps = predict_maps[:,1,:,:].view(B,D,H,W) # (1,2,3,*) -> (1, 3, 250, 250)
                    masks = masks.view(B, D,-1)

                    for lkh in range(likelihoodMaps.shape[1]):
                        save_prediction_likelihood(likelihoodMaps[:,lkh,:,:], str(batch)+'_{0}'.format(lkh), i, image_save_path) 
                        save_gt(masks.view(B,D,H,W)[:,lkh,:,:], str(batch)+'_{0}'.format(lkh), i, image_save_path)
                with torch.no_grad():
                    total_val_loss = total_val_loss + loss_fun(predict_maps, masks.to(device)).cpu().item()
                    # print('out', predict_map.shape) # (1, 2, 1250, 1250)
                    pred_class = torch.argmax(predict_maps, dim=1).float()  # (1, 3, 1250 * 1250)
                    # pred_class = likelihoodMap > 0.8                            
                    acc_val = accuracy_check(masks.cpu(), pred_class.cpu())
                    for lkh in range(pred_class.shape[1]):
                        im_name = str(batch)+'_{0}'.format(lkh) 
                        pred_msk = save_prediction_image(pred_class.view(B,D,H,W)[:,lkh,:,:], im_name, i, image_save_path)
                    total_val_acc += acc_val

            val_acc, val_loss = total_val_acc / (batch + 1), total_val_loss / (batch + 1)
            print('Val loss:', val_loss, "val acc:", val_acc)

            values = [i + 1, train_loss, train_acc, val_loss, val_acc]
            export_history(header, values, save_dir, save_file_name)

            # if (i + 1) % 10 == 0:  # save model every 10 epoch
            #     save_models(models, model_save_dir, i + 1, SLICES_COLLECT)        

if __name__ == "__main__":
    pretrain_epoch = 100  # 2000
    topo_epoch = 110
    COMPLETE_TRAIN = False
    LR = 0.0001
    DATASET = 'allen'
    SLICES_COLLECT = [3]
    modelName = '3'
    header = ['epoch', 'train loss', 'train acc', 'val loss', 'val acc']
    save_file_name = DATASET + "_2d_cross1/history{0}/history_Valid.csv".format(modelName)
    save_dir = DATASET + "_2d_cross1/history{0}/".format(modelName)

    # Saving images and models directories
    model_save_dir = DATASET + "_2d_cross1/history{0}/saved_models3".format(modelName)
    image_save_path = DATASET + "_2d_cross1/history{0}/result_images3".format(modelName)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = 'train_' + DATASET 
    # dataset_cache = 'data_cache/{0}_dataset_cache'.format(DATASET)
    dataset_cache = 'data_cache/fix_dataset_cache_3d_ISBI13'

    # trainDataset, validDataset = get_dataset(dataset_path, dataset_cache, SLICES_COLLECT)
    trainDataset, validDataset = get_3d_dataset(dataset_path, dataset_cache, SLICES_COLLECT)

    train_load = torch.utils.data.DataLoader(dataset=trainDataset, num_workers=6, batch_size=8, shuffle=True)
    val_load = torch.utils.data.DataLoader(dataset=validDataset, num_workers=6, batch_size=1, shuffle=False)

    train3D()
    # if COMPLETE_TRAIN:
    #     models = multiModel(SLICES_COLLECT)
    #     train()
        

    # elif os.path.exists(save_dir):
    #     print('loading pre_train model from ' + model_save_dir)
    #     models = load_models(model_save_dir, SLICES_COLLECT)
    #     topo_train(models)
        
    # else:
    #     models = multiModel(SLICES_COLLECT)
    #     pre_train()
    
    


    



    
"""
# Test
print("generate test prediction")
test_model("../history/RMS/saved_models/model_epoch_440.pwf",
           test_load, 440, "../history/RMS/result_images_test")
"""
