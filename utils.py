import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch


class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, smiles=None, smi_embedding_matrix=None):

        #  root是保存预处理数据所必需的，默认值为'/ tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # 基准数据集，默认='davis'
        self.dataset = dataset
        # 在self.processed_dir文件夹中找到的文件路径，以跳过处理。
        # self.processed_paths[0]：['data\\processed\\kiba_train.pt']
        if os.path.isfile(self.processed_paths[0]):
            print(
                'Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print(
                'Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y, smile_graph, smiles, smi_embedding_matrix)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # 定制处理方法以适应药物目标亲和力预测的任务
    # Inputs:
    # XD - list of SMILES, XT: 编码目标列表（分类或单目标），
    # Y:   标签列表 (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, y, smile_graph, smiles, smi_embedding_matrix):
        assert (len(xd) == len(xt) and len(xt) == len(
            y)),             """这三个列表的长度必须相同！"""
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smile = xd[i]
            target = xt[i]
            labels = y[i]
            # 使用rdkit将SMILES转换为分子表示
            c_size, features, edge_index = smile_graph[smile]
            smile_ = smiles[smile]
            # 为PyTorch Geometrics GCN算法准备好图形：
            # x=torch.Tensor(features):  [[78维]x原子数]
            # torch.LongTensor(edge_index).transpose(1, 0)：[[不定],[不定]]
            # torch.FloatTensor([labels]) ：[亲和力]
            # torch.LongTensor([target]) :[[1 X 1000]]
            # torch.LongTensor([c_size]):[原子数]
            # Data(edge_index=[2, 62], x=[29, 78], y=[1])
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(
                                    edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            GCNData.target = torch.LongTensor([target])  # 蛋白质序列的属性
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))  #
            GCNData.smile = torch.LongTensor([smile_])
            GCNData.smi_embedding = torch.FloatTensor(smi_embedding_matrix)
            # append graph, label and target sequence to data list
            # data_list:[Data(c_size=[1], edge_index=[2, 62], target=[1, 1000], x=[29, 78], y=[1]), Data(c_size=[1], edge_index=[2, 56], target=[1, 1000], x=[25, 78], y=[1]),....]
            data_list.append(GCNData)
            del GCNData

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        # data：Data(c_size=[4], edge_index=[2, 308], target=[4, 1000], x=[140, 78], y=[4])
        #             c_size:tensor([29, 25, 40, 46])
        #             edge_index：列数变多
        #             target：行变多了
        #             x：行变多了y
        #             y：列变多了
        # slices：{'x': [0, 29, 54, 94, 140], 'edge_index': [0, 62, 118, 206, 308], 'y': [0, 1, 2, 3, 4], 'target': [0, 1, 2, 3, 4], 'c_size': [0, 1, 2, 3, 4]}
        data, slices = self.collate(data_list)
        # 保存预处理的数据:
        torch.save((data, slices), self.processed_paths[0])


# 用来计算指标
def rmse(y, f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse


def mse(y, f):
    mse = ((y - f)**2).mean(axis=0)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci
