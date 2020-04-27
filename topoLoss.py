import matplotlib

matplotlib.use('Agg')
import time
import torch
import torch.nn as nn
import os
# import visdom
import random
from tqdm import tqdm as tqdm
import sys
from betti_compute import betti_number
# from TDFMain import *
from TDFMain_pytorch import *

steps = [-1, 1, 100, 150]
scales = [1, 1, 1, 1]
workers = 4
seed = time.time()
step_lr_n_epochs = 10

min_mae = 10000
min_epoch = 0
train_loss_list = []
epoch_list = []
test_error_list = []
epoch_loss = 0

topo_loss = 0
topo_grad = 0

# n = 0;
# topo_cp_map = np.zeros(et_dmap.shape);
n_fix = 0
n_remove = 0
pers_thd_lh = 0.03
pers_thd_gt = 0.03


def getPers(likelihoods, groundtruth):
    pd_lh_all, bcp_lh_all, dcp_lh_all, pd_gt_all, bcp_gt_all, dcp_gt_all, lh_pers_all, lh_pers_valid_all, gt_pers_all, gt_pers_valid_all = ([] for i in range(10))
    for likelihood in likelihoods:
        if torch.min(likelihood) == 1: continue
        if torch.max(likelihood) == 0: continue
        pd_lh, bcp_lh, dcp_lh = compute_persistence_2DImg_1DHom_lh(likelihood)

        if (pd_lh.shape[0] > 0):
            lh_pers = pd_lh[:, 1] - pd_lh[:, 0]
            lh_pers_valid = lh_pers[np.where(lh_pers > pers_thd_lh)]
        else:
            lh_pers = np.array([])
            lh_pers_valid = np.array([])

        pd_gt, bcp_gt, dcp_gt = compute_persistence_2DImg_1DHom_gt(groundtruth)

        if (pd_gt.shape[0] > 0):  # number of critical points (n, 2)
            gt_pers = pd_gt[:, 1] - pd_gt[:, 0]
            gt_pers_valid = gt_pers[np.where(gt_pers > pers_thd_gt)]
        else:
            gt_pers = np.array([])
            gt_pers_valid = np.array([])
        pd_lh_all.append(pd_lh)
        bcp_lh_all.append(bcp_lh) 
        dcp_lh_all.append(dcp_lh) 
        pd_gt_all.append(pd_gt) 
        bcp_gt_all.append(bcp_gt) 
        dcp_gt_all.append(dcp_gt) 
        lh_pers_all.append(lh_pers) 
        lh_pers_valid_all.append(lh_pers_valid) 
        gt_pers_all.append(gt_pers) 
        gt_pers_valid_all.append(gt_pers_valid)
    pd_lh_all = np.array([row for rows in pd_lh_all for row in rows])
    bcp_lh_all = np.array([row for rows in bcp_lh_all for row in rows])
    dcp_lh_all = np.array([row for rows in dcp_lh_all for row in rows])
    pd_gt_all = np.array([row for rows in pd_gt_all for row in rows])
    bcp_gt_all = np.array([row for rows in bcp_gt_all for row in rows])
    dcp_gt_all = np.array([row for rows in dcp_gt_all for row in rows])
    lh_pers_all = np.array([row for rows in lh_pers_all for row in rows])
    lh_pers_valid_all = np.array([row for rows in lh_pers_valid_all for row in rows])
    gt_pers_all = np.array([row for rows in gt_pers_all for row in rows])
    gt_pers_valid_all = np.array([row for rows in gt_pers_valid_all for row in rows])

    return pd_lh_all, bcp_lh_all, dcp_lh_all, pd_gt_all, bcp_gt_all, dcp_gt_all, lh_pers_all, lh_pers_valid_all, gt_pers_all, gt_pers_valid_all

def getTopoLoss(likelihoodMaps, binaryPredict, masks, device, likelihoodMap_final):
    topo_size = 65
    gt_dmap = masks.to(device)
    et_dmap = likelihoodMap_final
    n_fix = 0
    n_remove = 0
    topo_cp_weight_map = np.zeros(et_dmap.shape)
    topo_cp_ref_map = np.zeros(et_dmap.shape)
    allWindows = 1
    inWindows = 1

    for y in range(0, gt_dmap.shape[0], topo_size):
        for x in range(0, gt_dmap.shape[1], topo_size):
            likelihoodAll = []
            allWindows = allWindows + 1
            likelihood = et_dmap[y:min(y + topo_size, gt_dmap.shape[0]),
                         x:min(x + topo_size, gt_dmap.shape[1])]
            binary = binaryPredict[y:min(y + topo_size, gt_dmap.shape[0]),
                         x:min(x + topo_size, gt_dmap.shape[1])]           
            groundtruth = gt_dmap[y:min(y + topo_size, gt_dmap.shape[0]),
                          x:min(x + topo_size, gt_dmap.shape[1])]
            for likelihoodMap in likelihoodMaps:
                likelihoodAll.append(likelihoodMap[y:min(y + topo_size, gt_dmap.shape[0]),
                         x:min(x + topo_size, gt_dmap.shape[1])])

            # print('likelihood', likelihood.shape, 'groundtruth', groundtruth.shape, 'binaryPredict', binary.shape)
            predict_betti_number = betti_number(binary)
            groundtruth_betti_number = betti_number(groundtruth)
            # print(predict_betti_number, groundtruth_betti_number)

            if(torch.min(likelihood) == 1 or torch.max(likelihood) == 0): continue
            if (torch.min(groundtruth) == 1 or torch.max(groundtruth) == 0): continue
            if groundtruth_betti_number == 0: continue
            if all( torch.min(lkhd) == 1 for lkhd in likelihoodAll): continue
            if(abs(predict_betti_number - groundtruth_betti_number) / groundtruth_betti_number) < 0.4:
                continue           
            if (len(likelihood.shape) < 2 or len(groundtruth.shape) < 2):
                continue
            print('row: ', y, 'col: ', x)
            inWindows = inWindows + 1

            pd_lh, bcp_lh, dcp_lh, pd_gt, bcp_gt, dcp_gt, lh_pers, lh_pers_valid, gt_pers, gt_pers_valid = getPers(likelihoodAll, groundtruth)
            if (pd_lh.shape[0] > gt_pers_valid.shape[0]):
                force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(pd_lh, pd_gt)
                n_fix += len(idx_holes_to_fix)
                n_remove += len(idx_holes_to_remove)
                if (len(idx_holes_to_fix) > 0 or len(idx_holes_to_remove) > 0):
                    # print('#####################################################################')
                    # bcp_lh = bcp_lh + padwidth;
                    # dcp_lh = dcp_lh + padwidth;
                    for hole_indx in idx_holes_to_fix:
                        # print('in loop fix')
                        # print('hole_indx=',hole_indx)
                        # print(y+int(bcp_lh[hole_indx][0]) < et_dmap.shape[2] and x+int(bcp_lh[hole_indx][1]) < et_dmap.shape[3])
                        # print(x, y, int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1]), et_dmap.shape[2], et_dmap.shape[3])
                        # print(y+int(dcp_lh[hole_indx][0]) < et_dmap.shape[2] and x+int(dcp_lh[hole_indx][1]) < et_dmap.shape[3])
                        # print(x, y, int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1]), et_dmap.shape[2], et_dmap.shape[3])
                        # if(y+int(bcp_lh[hole_indx][0]) < et_dmap.shape[2] and x+int(bcp_lh[hole_indx][1]) < et_dmap.shape[3]):

                        if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]):
                            topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 1 # push birth to 0 i.e. min birth prob or likelihood
                            topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 0
                        # if(y+int(dcp_lh[hole_indx][0]) < et_dmap.shape[2] and x+int(dcp_lh[hole_indx][1]) < et_dmap.shape[3]):
                        if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                                dcp_lh[hole_indx][1])] = 1  # push death to 1 i.e. max death prob or likelihood
                            topo_cp_ref_map[ y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 1
                    for hole_indx in idx_holes_to_remove:
                        # print('in loop remove')
                        # print('hole_indx=',hole_indx)
                        # print(y+int(bcp_lh[hole_indx][0]) < et_dmap.shape[2] and x+int(bcp_lh[hole_indx][1]) < et_dmap.shape[3])
                        # print(x, y, int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1]), et_dmap.shape[2], et_dmap.shape[3])
                        # print(y+int(dcp_lh[hole_indx][0]) < et_dmap.shape[2] and x+int(dcp_lh[hole_indx][1]) < et_dmap.shape[3])
                        # print(x, y, int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1]), et_dmap.shape[2], et_dmap.shape[3])
                        # if(y+int(bcp_lh[hole_indx][0]) < et_dmap.shape[2] and x+int(bcp_lh[hole_indx][1]) < et_dmap.shape[3]):
                        if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                                bcp_lh[hole_indx][1])] = 1  # push birth to death  # push to diagonal
                            # if(int(dcp_lh[hole_indx][0]) < likelihood.shape[0] and int(dcp_lh[hole_indx][1]) < likelihood.shape[1]):
                            if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                                0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                                    likelihood.shape[1]):
                                topo_cp_ref_map[ y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = \
                                    likelihood[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])]
                            else:
                                topo_cp_ref_map[ y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 1
                                # if(y+int(dcp_lh[hole_indx][0]) < et_dmap.shape[2] and x+int(dcp_lh[hole_indx][1]) < et_dmap.shape[3]):
                        if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                                dcp_lh[hole_indx][1])] = 1  # push death to birth # push to diagonal
                            # if(int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]):
                            if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                                0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                                    likelihood.shape[1]):
                                topo_cp_ref_map[ y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = \
                                    likelihood[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])]
                            else:
                                topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 0

    topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).to(device)
    topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).to(device)
    loss_topo = (((et_dmap * topo_cp_weight_map) - topo_cp_ref_map) ** 2).sum()
    print("not scape per: ", inWindows / allWindows, 'loss_topo',loss_topo)

    return loss_topo, 1 - (inWindows / allWindows)

    # a = np.array([[1,2],[3,4]])
    # b = np.array([[3,76]])
    # c = [a, b]
    # print(np.array([row for rows in c for row in rows]))
    # testing phase
    # model.eval()
    # mae = 0
    # for i, (img, gt_dmap) in enumerate(tqdm(test_loader)):
    #     img = img.to(device)
    #     gt_dmap = gt_dmap.to(device)
    #     # forward propagation
    #     et_dmap = model(img)
    #     mae += abs(et_dmap.data.sum() - gt_dmap.data.sum()).item()
    #     del img, gt_dmap, et_dmap
    # if mae / len(test_loader) < min_mae:
    #     min_mae = mae / len(test_loader)
    #     min_epoch = epoch
    # test_error_list.append(mae / len(test_loader))
    # print("epoch:" + str(epoch) + " error:" + str(mae / len(test_loader)) + " min_mae:" + str(
    #     min_mae) + " min_epoch:" + str(min_epoch))
    # print("epoch_loss:" + str(epoch_loss / len(train_loader)))
    # print("epoch_loss_topo:" + str(epoch_loss_topo / len(train_loader)))
    # print("epoch_loss_topo_total:" + str(epoch_loss_topo_total / len(train_loader)))
    # print("epoch_loss_mse:" + str(epoch_loss_mse / len(train_loader)))

    # sys.stdout.flush()

    # import time

    # print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    # sys.stdout.flush()
