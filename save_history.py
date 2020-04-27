import os
import csv
import torch
from model import U_Net

def export_history(header, value, folder, file_name):
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_existence = os.path.isfile(file_name)
    if not file_existence:
        file = open(file_name, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(value)
    else:
        file = open(file_name, 'a', newline='')
        writer = csv.writer(file)
        writer.writerow(value)
    file.close()

def export_topo_train(epoch, batch, lamda, loss, loss_ce, topoLoss, save_dir, escapes):
    file_existence = os.path.isfile('{0}history_topo.csv'.format(save_dir))
    header = ['epoch', 'batch', 'lambda', 'loss_total', 'loss_ce', 'topoLoss', 'escapes']
    value = [epoch + 1, batch, lamda, loss, loss_ce, topoLoss, escapes]
    if not file_existence:
        file = open('{0}history_topo.csv'.format(save_dir), 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(value)
    else:
        file = open('{0}history_topo.csv'.format(save_dir), 'a', newline='')
        writer = csv.writer(file)
        writer.writerow(value)
    file.close()

def save_models(models, path, epoch, SLICES_COLLECT):
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(len(SLICES_COLLECT)):
        torch.save(models[i], path + "/model_epoch_{0}_{1}.pwf".format(epoch, i+1))

def load_models(model_save_dir, SLICES_COLLECT):
    models = []
    for i in range(len(SLICES_COLLECT)):
        print(model_save_dir + "/model_epoch_30_{0}.pwf".format(i+1))
        checkpoint = torch.load(model_save_dir + "/model_epoch_30_{0}.pwf".format(i+1)) 
        model = U_Net(in_channels=SLICES_COLLECT[i], out_channels = 32)
        model.load_state_dict(checkpoint.module.state_dict())
        models.append(model)
    return models
