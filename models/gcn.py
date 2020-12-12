import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, global_max_pool as gmp


# GCN based model


class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128, num_features_xd=78, num_features_xt=25, output_dim=128, dropout=0.2, data=None):

        super(GCNNet, self).__init__()

        # SMILES graph 分支
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.embed_smile = nn.Embedding(100, 100)
        self.embed_smile.weight = nn.Parameter(data.smi_embedding)
        self.embed_smile.weight.requires_grad = True

        self.fc1_xs = nn.Linear(100*200, output_dim)
        self.W_rnn = nn.GRU(bidirectional=True, num_layers=1,
                            input_size=100, hidden_size=100)

        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.rnn = nn.LSTM(
            embed_dim, 32, 3, batch_first=True, bidirectional=True)
        # self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)

        # 连接层
        self.fc1 = nn.Linear(128+200+192, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def attention_net(self, lstm_output, final_state):
        # hidden : [batch_size, n_hidden * num_directions(=2), 3(=n_layer)]
        hidden = final_state.view(-1, 32 * 2, 3)
        # print('hidden')
        # print(hidden.shape)
        # attn_weights : [batch_size, n_step]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        # print('attn_weights')
        # print(attn_weights.shape)
        soft_attn_weights = F.softmax(attn_weights, 1)
        # print('soft_attn_weights')
        # print(soft_attn_weights.shape)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights)
        # print('context')
        # print(context.shape)
        # context : [batch_size, n_hidden * num_directions(=2)]
        return context, soft_attn_weights.data.cpu().numpy()

    def gru(self, xs):
        # print(xs.shape)  # torch.Size([2, 128])
        xs, h = self.W_rnn(xs)
        # print(xs.shape)  # torch.Size([2, 100, 200])

        xs = torch.relu(xs)
        # print(xs.shape)  # torch.Size([2, 100, 200])
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        # print(xs.shape)  # torch.Size([2, 100, 200])
        return torch.mean(xs, 1)  # torch.Size([2,200])

    def forward(self, data):
        # get graph input 获取药物输入
        x, edge_index, batch = data.x, data.edge_index, data.batch
        smile = data.smile
        # get protein input 获取蛋白质的输入
        target = data.target
        '''药物的图处理模型'''
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       # 全局最大池化层
        # print(x.shape)  # torch.Size([2, 312])

        x = self.relu(self.fc_g1(x))

        # print(x.shape) torch.Size([2, 1024])
        x = self.dropout(x)
        x = self.fc_g2(x)

        # print(x.shape)    torch.Size([2, 128])
        x = self.dropout(x)
        # print('x')
        # print(x.shape)    #torch.Size([2, 128])

        smile_vectors = self.embed_smile(smile)
        after_smile_vectors = self.gru(smile_vectors)

        '''对于蛋白质的处理'''
        embedded_xt = self.embedding_xt(target)
        h_0 = Variable(torch.zeros(6, len(target), 32).cuda())
        c_0 = Variable(torch.zeros(6, len(target), 32).cuda())
        output, (hn, cn) = self.rnn(embedded_xt, (h_0, c_0))

        attn_output, attention = self.attention_net(output, hn)
        xt = attn_output.view(len(target), -1)
        # print(xt.shape)

        # concat
        xc = torch.cat((x, after_smile_vectors, xt), 1)

        # 添加全连接层
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        out = self.out(xc)

        return out
