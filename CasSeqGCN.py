#encoding:utf-8
import torch
import numpy as np
import utils
import glob
from tqdm import tqdm
import json
import copy
import os
import random
from sklearn import preprocessing
from layers import my_GCN,my_LSTM, dens_Net, dynamic_routing


class CasSeqGCN(torch.nn.Module):
    def __init__(self, args, number_of_features, number_of_nodes):
        """
        :param args:
        :param number_of_nodes: Number of vertex
        :param number_of_out_channel: Number of GCN out channel
        """
        super(CasSeqGCN, self).__init__()
        self.args = args
        self.number_of_features = number_of_features
        self.number_of_nodes = number_of_nodes
        self._setup_layers()

    def _setup_GCN_layers(self):

        self.GCN_layers = my_GCN(in_channels=self.number_of_features,
                                 out_channels=self.args.gcn_out_channel,
                                 filters_1=self.args.gcn_filters_1,
                                 filters_2=self.args.gcn_filters_2,
                                 dropout=self.args.gcn_dropout)

    def _setup_LSTM_layers(self):
        self.LSTM_layers = my_LSTM(lstm_inputsize=self.args.capsule_out_dim,
                                   lstm_hiddensize=self.args.lstm_hiddensize,
                                   lstm_layers=self.args.lstm_layers,
                                   lstm_dropout=self.args.lstm_dropout,
                                   )
    def _setup_dens_layers(self):
        self.dens_layers = dens_Net(dens_inputsize=self.args.lstm_hiddensize,
                                    dens_hiddensize=self.args.dens_hiddensize,
                                    dens_dropout=self.args.dens_dropout,
                                    dens_outputsize=self.args.dens_outsize
                                    )
    def _setup_dynamic_routing_layers(self):
        self.dynamic_routing = dynamic_routing(in_dim=32, out_dim=32
        )

    def _setup_layers(self):
        self._setup_GCN_layers()
        self._setup_LSTM_layers()
        self._setup_dens_layers()
        self._setup_dynamic_routing_layers()

    def forward(self, data):
        hidden_representations = []
        for sub_graph in data:
            features = sub_graph['features']
            edges = sub_graph['edges']
            gcn_representation = torch.nn.functional.relu(self.GCN_layers(edges, features))
            hidden_representations.append(gcn_representation)
        hidden_representations = torch.cat(tuple(hidden_representations))
        sub_graph_representation = hidden_representations.view(-1, self.number_of_nodes, self.args.gcn_out_channel)

        sub_graph_representation = self.dynamic_routing(sub_graph_representation)




        sub_graph_representation = sub_graph_representation.unsqueeze(dim=0)
        graph_representation = self.LSTM_layers(sub_graph_representation)
        prediction = self.dens_layers(graph_representation)
        prediction = torch.nn.functional.relu(prediction)
        prediction = prediction.squeeze(-1)

        return graph_representation, prediction


class CasSeqGCNTrainer(object):
    def __init__(self, args):
        self.args = args
        self.setup_model()

    def enumerate_unique_labels_and_targets(self):
        """
        Enumerating the features and targets.
        """
        print("\nEnumerating feature and target values.\n")
        ending = "*.json"

        self.graph_paths = sorted(glob.glob(self.args.graph_folder + ending), key = os.path.getmtime)

        features = set()
        data_dict = dict()
        for path in tqdm(self.graph_paths):
            data = json.load(open(path))
            data_dict = data
            for i in range(0, len(data) - self.args.sub_size): 
                graph_num = 'graph_' + str(i)
                features = features.union(set(data[graph_num]['labels'].values()))


        self.number_of_nodes = self.args.number_of_nodes
        self.feature_map = utils.create_numeric_mapping(features)
        self.number_of_features = len(self.feature_map)


    def setup_model(self):
        self.enumerate_unique_labels_and_targets()
        self.model = CasSeqGCN(self.args, self.number_of_features + self.args.number_of_hand_features, self.number_of_nodes)

    def create_batches(self):
        N = len(self.graph_paths)
        train_start, valid_start, test_start = \
            0, int(N * self.args.train_ratio), int(N * (self.args.train_ratio + self.args.valid_ratio))
        train_graph_paths = self.graph_paths[0:valid_start]
        valid_graph_paths = self.graph_paths[valid_start:test_start]
        test_graph_paths = self.graph_paths[test_start: N]
        self.train_batches, self.valid_batches, self.test_batches = [], [], []
        for i in range(0, len(train_graph_paths), self.args.batch_size):
            self.train_batches.append(train_graph_paths[i:i+self.args.batch_size])
        for j in range(0, len(valid_graph_paths), self.args.batch_size):
            self.valid_batches.append(valid_graph_paths[j:j+self.args.batch_size])
        for k in range(0, len(test_graph_paths), self.args.batch_size):
            self.test_batches.append(test_graph_paths[k:k+self.args.batch_size])

    def create_data_dictionary(self, edges, features):
        """
        creating a data dictionary
        :param target: target vector
        :param edges: edge list tensor
        :param features: feature tensor
        :return:
        """
        to_pass_forward = dict()
        to_pass_forward["edges"] = edges
        to_pass_forward["features"] = features
        return to_pass_forward

    def create_target(self, data):
        """
        Target createn based on data dicionary.
        :param data: Data dictionary.
        :return: Target size
        """
        return  torch.tensor([data['activated_size']])

    def create_edges(self,data):
        """
        Create an edge matrix.
        :param data: Data dictionary.
        :return : Edge matrix.
        """
        self.nodes_map = [str(nodes_id) for nodes_id in data['nodes']] 
        edges = [[self.nodes_map.index(str(edge[0])), self.nodes_map.index(str(edge[1]))] for edge in data["edges"]]
        edges = edges + [[self.nodes_map.index(str(edge[1])), self.nodes_map.index(str(edge[0]))] for edge in data["edges"]]
        return torch.t(torch.LongTensor(edges))

    def create_features(self, data):
        """
        Create feature matrix.
        :param data: Data dictionary.
        :return features: Matrix of features.
        """
        features = np.zeros((self.number_of_nodes, self.number_of_features + 2))
        node_indices = [self.nodes_map.index(node) for node in data['labels'].keys()]
        feature_indices = [self.feature_map[label] for label in data['labels'].values()]
        features[node_indices, feature_indices] = 1.0
        for node_index in node_indices:
            features[node_index, self.number_of_features] = self.in_degree[str(self.nodes_map[node_index])]
            features[node_index, self.number_of_features+1] = self.out_degree[str(self.nodes_map[node_index])]
        features = torch.FloatTensor(features)
        return features


    def create_input_data(self, path):
        """
        Creating tensors and a data dictionary with Torch tensors.
        :param path: path to the data JSON.
        :return to_pass_forward: Data dictionary.
        """
        data = json.load(open(path))
        to_pass_forward = []
        activated_size = self.create_target(data)
        edges = self.create_edges(data['graph_info'])
        self.in_degree = data['graph_info']['in_degree']
        self.out_degree = data['graph_info']['out_degree']
        for i in range(0, len(data) - self.args.sub_size):
            graph_num = 'graph_' + str(i)
            features = self.create_features(data[graph_num])
            to_pass_forward.append(self.create_data_dictionary(edges, features))
        return to_pass_forward, activated_size
    def create_train_data(self):
        train_x, train_y =  [], []
        for train_batch in self.train_batches:
            train_x_tmp, train_y_tmp = [], []
            for train_path in train_batch:
                train_data, train_target = self.create_input_data(train_path)
                train_x_tmp.append(train_data)
                train_y_tmp.append(train_target)
            train_x.append(copy.deepcopy(train_x_tmp))
            train_y.append(copy.deepcopy(train_y_tmp))
        return train_x, train_y
    def create_valid_data(self):
        valid_x, valid_y = [], []
        for valid_batch in self.valid_batches:
            valid_x_tmp, valid_y_tmp =  [], []
            for valid_path in  valid_batch:
                valid_data, valid_target = self.create_input_data(valid_path)
                valid_x_tmp.append(valid_data)
                valid_y_tmp.append(valid_target)
            valid_x.append(copy.deepcopy(valid_x_tmp))
            valid_y.append(copy.deepcopy(valid_y_tmp))
        return valid_x, valid_y
    def create_test_data(self):
        test_x, test_y = [], []
        for test_batch in self.test_batches:
            test_x_tmp, test_y_tmp = [], []
            for test_path in test_batch:
                test_data, test_target = self.create_input_data(test_path)
                test_x_tmp.append(test_data)
                test_y_tmp.append(test_target)
            test_x.append(copy.deepcopy(test_x_tmp))
            test_y.append(copy.deepcopy(test_y_tmp))
        return test_x, test_y


    def fit(self):
        """
        Training a model on the training set.
        """
        print('\nTraining started.\n')
        self.model.train()
        self.create_batches()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        train_x_batches, train_y_batches = self.create_train_data()
        valid_x_batches, valid_y_batches = self.create_valid_data()
        train_graph_represent, valid_graph_represent, test_graph_represent = [], [], []
        for epoch in range(self.args.epochs):
            # self.create_batches()
            losses = 0.
            average_loss = 0.
            for step, (train_x_batch, train_y_batch) in enumerate(zip(train_x_batches, train_y_batches)):
                loss = 0.
                loss_back = 0.
                optimizer.zero_grad()
                prediction_tensor = torch.tensor([])
                target_tensor = torch.tensor([])
                for k, (train_x, train_y) in enumerate(zip(train_x_batch, train_y_batch)):
                    graph_representation, prediction = self.model(train_x)
                    prediction_tensor = torch.cat((prediction_tensor, prediction.float()), 0)
                    target_tensor = torch.cat((target_tensor, torch.log2(train_y.float() + 1)), 0)
                loss = torch.nn.functional.mse_loss(target_tensor,prediction_tensor)
                loss.backward()
                optimizer.step()
                losses = losses + loss.item()
                average_loss = losses / (step + 1)
            print('CasSeqGCN train MSLE loss in ', epoch + 1, ' epoch = ', average_loss)
            print('\n')

            if (epoch + 1) % self.args.check_point == 0:
                print('epoch ',epoch + 1, ' evaluating.')
                self.evaluation(valid_x_batches, valid_y_batches)
                self.test()

    def evaluation(self, valid_x_batches, valid_y_batches):
        self.model.eval()
        losses = 0.
        average_loss = 0.
        for step, (valid_x_batch, valid_y_batch) in enumerate(zip(valid_x_batches, valid_y_batches)):
            loss = 0.
            prediction_tensor = torch.tensor([])
            target_tensor = torch.tensor([])
            for (valid_x, valid_y) in zip(valid_x_batch, valid_y_batch):
                graph_representation, prediction = self.model(valid_x)
                prediction_tensor = torch.cat((prediction_tensor, prediction.float()), 0)
                target_tensor = torch.cat((target_tensor, torch.log2(valid_y.float() + 1)), 0)
            loss = torch.nn.functional.mse_loss(target_tensor, prediction_tensor)
            losses = losses + loss.item()
            average_loss = losses / (step + 1)
        print('#####CasSeqGCN valid MSLE loss in this epoch = ', average_loss)
        print('\n')

    def test(self):
        print("\n\nScoring.\n")
        self.model.eval()
        test_x_batches, test_y_batches = self.create_test_data()
        losses = 0.
        average_loss = 0.
        for step, (test_x_batch, test_y_batch) in enumerate(zip(test_x_batches, test_y_batches)):
            loss = 0.
            prediction_tensor = torch.tensor([])
            target_tensor = torch.tensor([])
            for (test_x, test_y) in zip(test_x_batch, test_y_batch):
                graph_representation, prediction = self.model(test_x)
                prediction_tensor = torch.cat((prediction_tensor, prediction.float()), 0)
                target_tensor = torch.cat((target_tensor, torch.log2(test_y.float() + 1)), 0)
            loss = torch.nn.functional.mse_loss(target_tensor, prediction_tensor)
            losses = losses + loss.item()
            average_loss = losses / (step + 1)
        print('#####CasSeqGCN test MSLE loss = ', average_loss)
        print('\n')