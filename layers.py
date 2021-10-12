#encoding:utf-8


import torch
from torch_geometric.nn import GCNConv
from torch.autograd import Variable



class my_GCN(torch.nn.Module):

    def __init__(self,in_channels, out_channels,filters_1, filters_2, dropout, bais=True):
        """
        GCN function
        :param args:  Arguments object.
        :param in_channel: Nodes' input feature dimensions
        :param out_channel: Nodes embedding dimension
        :param bais:
        """
        super(my_GCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters_1 = filters_1
        self.filters_2 = filters_2
        self.dropout = dropout
        self.setup_layers()


    def setup_layers(self):
        self.convolution_1 = GCNConv(self.in_channels, self.filters_1)
        self.convolution_2 = GCNConv(self.filters_1, self.filters_2)
        self.convolution_3 = GCNConv(self.filters_2, self.out_channels)


    def forward(self, edge_indices, features):
        """
        making convolution
        :param edge_indices: 2 * edge_number
        :param features: N * feature_size
        :return:
        """
        features = self.convolution_1(features, edge_indices)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = self.convolution_2(features, edge_indices)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_indices)
        return features

class dynamic_routing(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(dynamic_routing,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = torch.nn.Parameter(torch.randn(1,in_dim,out_dim))

    def forward(self,x):
        num_nodes = x.size(1)  # sub_graph size
        batch_size = x.size(0)
        W = torch.cat([self.W] * batch_size, dim=0)
        representation = torch.matmul(x, W)
        r_sum = torch.sum(representation, dim=-1, keepdim=False)
        b = torch.zeros([batch_size, num_nodes])
        b = Variable(b)
        one = torch.ones_like(r_sum)
        zero = torch.zeros_like(r_sum)
        label = torch.clone(r_sum)
        label = torch.where(label == 0, one, zero)
        b.data.masked_fill_(label.bool(), -float('inf'))
        num_iterations = 3
        for i in range(num_iterations):
            c = torch.nn.functional.softmax(b, dim=-1)
            weight_coeff = c.unsqueeze(dim=1)
            representation_global = torch.matmul(weight_coeff, representation)
            representation_global_all = torch.cat([representation_global] * num_nodes, dim=1)
            representation_similarity = torch.nn.functional.cosine_similarity(representation, representation_global_all, dim=-1)
            representation_similarity.data.masked_fill_(label.bool(), -float('inf'))
            b = representation_similarity
        return representation_global.squeeze(dim=1)


class my_LSTM(torch.nn.Module):
    def __init__(self, lstm_inputsize, lstm_hiddensize, lstm_layers, lstm_dropout):
        super(my_LSTM, self).__init__()
        self.lstm_inputsize = lstm_inputsize
        self.lstm_hiddensize = lstm_hiddensize
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.setup_layers()

    def setup_layers(self):
        self.lstm = torch.nn.LSTM(
            input_size = self.lstm_inputsize,
            hidden_size = self.lstm_hiddensize,
            num_layers = self.lstm_layers,
            batch_first=True,
            dropout=(0 if self.lstm_layers == 1 else self.lstm_dropout),
            bidirectional=False
        )

    def forward(self, input):
        out, (h_n, c_n) = self.lstm(input)
        return out[:, -1, :]

class dens_Net(torch.nn.Module):
    def __init__(self,dens_hiddensize, dens_dropout,  dens_inputsize, dens_outputsize):
        super(dens_Net, self).__init__()
        self.inputsize = dens_inputsize
        self.dens_hiddensize = dens_hiddensize
        self.dens_dropout = dens_dropout
        self.outputsize = dens_outputsize
        self.setup_layers()

    def setup_layers(self):
        self.dens_net = torch.nn.Sequential(
            torch.nn.Linear(self.inputsize, self.dens_hiddensize),
            torch.nn.Dropout(p=self.dens_dropout),
            torch.nn.Linear(self.dens_hiddensize, self.dens_hiddensize),
            torch.nn.Dropout(p=self.dens_dropout),
            torch.nn.Linear(self.dens_hiddensize, self.outputsize)
        )

    def forward(self, x):
        return self.dens_net(x)
