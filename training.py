import numpy as np
import pandas as pd
import sys
import os
from random import shuffle
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *


# training function at each epoch


def train(model, device, train_loader, optimizer, epoch, scheduler):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    # 0 Batch(batch=[40], c_size=[1], edge_index=[2, 88], target=[1, 1000], x=[40, 78], y=[1])
    # 1 Batch(batch=[25], c_size=[1], edge_index=[2, 56], target=[1, 1000], x=[25, 78], y=[1])
    # 2 Batch(batch=[29], c_size=[1], edge_index=[2, 62], target=[1, 1000], x=[29, 78], y=[1])
    # 3 Batch(batch=[46], c_size=[1], edge_index=[2, 102], target=[1, 1000], x=[46, 78], y=[1])
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()  # 清除梯度
        output = model(data)
        # data.y.view(-1, 1):tensor([[5.]])
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()  # 优化器调度
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx *
                                                                           len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx /
                                                                           len(train_loader),
                                                                           loss.item()))


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat(
                (total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


datasets = [['davis', 'kiba'][int(sys.argv[1])]]
# datasets = [['davis', 'kiba'][0]]
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
# modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][3]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = ["cuda:0", "cuda:1"][0]
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset)
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset + '_train')
        test_data = TestbedDataset(root='data', dataset=dataset + '_test')
        # make data PyTorch mini-batch processing ready
        # 准备好数据PyTorch迷你批处理
        train_loader = DataLoader(
            train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(
            test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # 训练模型
        device = torch.device(
            cuda_name if torch.cuda.is_available() else "cpu")

        model = modeling(data=train_data[0]).to(device)

        loss_fn = nn.MSELoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # 定义Adamw优化器
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        # 优化器调度
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                    num_training_steps=len(train_loader) * NUM_EPOCHS)
        best_mse = 1000
        best_ci = 0
        best_epoch = -1
        model_file_name = 'model_' + model_st + '_' + dataset + '.model'
        result_file_name = 'result_' + model_st + '_' + dataset + '.csv'
        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch + 1,scheduler)
            G, P = predicting(model, device, test_loader)
            ret = [rmse(G, P), mse(G, P), pearson(
                G, P), spearman(G, P), ci(G, P)]
            if ret[1] < best_mse:
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name, 'w') as f:
                    f.write(','.join(map(str, ret)))
                best_epoch = epoch + 1
                best_mse = ret[1]
                best_ci = ret[-1]
                print('rmse improved at epoch ', best_epoch,
                      '; best_mse,best_ci:', best_mse, best_ci, model_st, dataset)
            else:
                print(ret[1], 'No improvement since epoch ', best_epoch,
                      '; best_mse,best_ci:', best_mse, best_ci, model_st, dataset)
