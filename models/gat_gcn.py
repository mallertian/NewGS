import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GCN-CNN based model


class GAT_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, data=None):

        super(GAT_GCN, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
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
            embed_dim, 32, 3, bidirectional=True, batch_first=True)

        # combined layers
        self.fc1 = nn.Linear(128+200+192, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

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

        return torch.mean(xs, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        smile = data.smile

        target = data.target
        # print('x shape = ', x.shape)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        smile_vectors = self.embed_smile(smile)
        after_smile_vectors = self.gru(smile_vectors)

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
