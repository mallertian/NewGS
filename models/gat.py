import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp

# GAT  model


class GATNet(torch.nn.Module):

    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, data=None):
        super(GATNet, self).__init__()

        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd,
                            heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        self.embed_smile = nn.Embedding(100, 100)
        self.embed_smile.weight = nn.Parameter(data.smi_embedding)
        self.embed_smile.weight.requires_grad = True

        self.fc1_xs = nn.Linear(100*200, output_dim)
        self.W_rnn = nn.GRU(bidirectional=True, num_layers=1,
                            input_size=100, hidden_size=100)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        # self.conv_xt1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        # self.fc_xt1 = nn.Linear(32*121, output_dim)
        self.rnn = nn.LSTM(
            embed_dim, 32, 3, bidirectional=True, batch_first=True)

        # combined layers
        self.fc1 = nn.Linear(128+200+192, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def attention_net(self, lstm_output, final_state):
        # hidden : [batch_size, n_hidden * num_directions(=2), 3(=n_layer)]
        hidden = final_state.view(-1, 32 * 2, 3)
        # attn_weights : [batch_size, n_step]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)

        soft_attn_weights = F.softmax(attn_weights, 1)

        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights)

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
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch
        smile = data.smile
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        smile_vectors = self.embed_smile(smile)
        after_smile_vectors = self.gru(smile_vectors)

        # protein input feed-forward:
        target = data.target
        embedded_xt = self.embedding_xt(target)
        h_0 = Variable(torch.zeros(6, len(target), 32).cuda())
        c_0 = Variable(torch.zeros(6, len(target), 32).cuda())
        output, (hn, cn) = self.rnn(embedded_xt, (h_0, c_0))

        attn_output, attention = self.attention_net(output, hn)
        xt = attn_output.view(len(target), -1)
        # concat
        xc = torch.cat((x, after_smile_vectors, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
