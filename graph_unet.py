import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import copy
import time
import math
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from os.path import join as pjoin

print('using torch', torch.__version__)

# Experiment parameters
parser = argparse.ArgumentParser(description='Graph Convolutional Networks')
parser.add_argument('-D', '--dataset', type=str, default='PROTEINS')
parser.add_argument('-M', '--model', type=str, default='gcn', choices=['gcn', 'unet', 'mgcn'])
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--lr_decay_steps', type=str, default='25,35', help='learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('-d', '--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('-f', '--filters', type=str, default='64,64,64', help='number of filters in each layer')
parser.add_argument('--n_hidden', type=int, default=0,
                    help='number of hidden units in a fully connected layer after the last conv layer')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
parser.add_argument('-t', '--threads', type=int, default=0, help='number of threads to load data')
parser.add_argument('--log_interval', type=int, default=10, help='interval (number of batches) of logging')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
parser.add_argument('--seed', type=int, default=111, help='random seed')
parser.add_argument('--shuffle_nodes', action='store_true', default=False, help='shuffle nodes for debugging')
parser.add_argument('-g', '--torch_geom', action='store_true', default=False, help='use PyTorch Geometric')
parser.add_argument('-a', '--adj_sq', action='store_true', default=False,
                    help='use A^2 instead of A as an adjacency matrix')
parser.add_argument('-s', '--scale_identity', action='store_true', default=False,
                    help='use 2I instead of I for self connections')
parser.add_argument('-v', '--visualize', action='store_true', default=False,
                    help='only for unet: save some adjacency matrices and other data as images')
parser.add_argument('-c', '--use_cont_node_attr', action='store_true', default=False,
                    help='use continuous node attributes in addition to discrete ones')

args = parser.parse_args()

if args.torch_geom:
    from torch_geometric.datasets import TUDataset

args.filters = list(map(int, args.filters.split(',')))
args.lr_decay_steps = list(map(int, args.lr_decay_steps.split(',')))

for arg in vars(args):
    print(arg, getattr(args, arg))

n_folds = 10  # 10-fold cross validation
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
rnd_state = np.random.RandomState(args.seed)


def split_ids(ids, folds=10):
    n = len(ids)
    stride = int(np.ceil(n / float(folds)))
    test_ids = [ids[i: i + stride] for i in range(0, n, stride)]
    assert np.all(
        np.unique(np.concatenate(test_ids)) == sorted(ids)), 'some graphs are missing in the test sets'
    assert len(test_ids) == folds, 'invalid test sets'
    train_ids = []
    for fold in range(folds):
        train_ids.append(np.array([e for e in ids if e not in test_ids[fold]]))
        assert len(train_ids[fold]) + len(test_ids[fold]) == len(
            np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == n, 'invalid splits'

    return train_ids, test_ids


if not args.torch_geom:
    # Unversal data loader and reader (can be used for other graph datasets from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)
    class GraphData(torch.utils.data.Dataset):
        def __init__(self,
                     datareader,
                     fold_id,
                     split):
            self.fold_id = fold_id
            self.split = split
            self.rnd_state = datareader.rnd_state
            self.set_fold(datareader.data, fold_id)

        def set_fold(self, data, fold_id):
            self.total = len(data['targets'])
            self.N_nodes_max = data['N_nodes_max']
            self.num_classes = data['num_classes']
            self.num_features = data['num_features']
            self.idx = data['splits'][fold_id][self.split]
            # use deepcopy to make sure we don't alter objects in folds
            self.labels = copy.deepcopy([data['targets'][i] for i in self.idx])
            self.adj_list = copy.deepcopy([data['adj_list'][i] for i in self.idx])
            self.features_onehot = copy.deepcopy([data['features_onehot'][i] for i in self.idx])
            print('%s: %d/%d' % (self.split.upper(), len(self.labels), len(data['targets'])))

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            # convert to torch
            return [torch.from_numpy(self.features_onehot[index]).float(),  # node_features
                    torch.from_numpy(self.adj_list[index]).float(),  # adjacency matrix
                    int(self.labels[index])]


    class DataReader():
        '''
        Class to read the txt files containing all data of the dataset
        '''

        def __init__(self,
                     data_dir,  # folder with txt files
                     rnd_state=None,
                     use_cont_node_attr=False,
                     # use or not additional float valued node attributes available in some datasets
                     folds=10):

            self.data_dir = data_dir
            self.rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
            self.use_cont_node_attr = use_cont_node_attr
            files = os.listdir(self.data_dir)
            data = {}
            nodes, graphs = self.read_graph_nodes_relations(
                list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0])
            data['features'] = self.read_node_features(list(filter(lambda f: f.find('node_labels') >= 0, files))[0],
                                                       nodes, graphs, fn=lambda s: int(s.strip()))
            data['adj_list'] = self.read_graph_adj(list(filter(lambda f: f.find('_A') >= 0, files))[0], nodes, graphs)
            data['targets'] = np.array(
                self.parse_txt_file(list(filter(lambda f: f.find('graph_labels') >= 0, files))[0],
                                    line_parse_fn=lambda s: int(float(s.strip()))))

            if self.use_cont_node_attr:
                data['attr'] = self.read_node_features(list(filter(lambda f: f.find('node_attributes') >= 0, files))[0],
                                                       nodes, graphs,
                                                       fn=lambda s: np.array(list(map(float, s.strip().split(',')))))

            features, n_edges, degrees = [], [], []
            for sample_id, adj in enumerate(data['adj_list']):
                N = len(adj)  # number of nodes
                if data['features'] is not None:
                    assert N == len(data['features'][sample_id]), (N, len(data['features'][sample_id]))
                n = np.sum(adj)  # total sum of edges
                assert n % 2 == 0, n
                n_edges.append(int(n / 2))  # undirected edges, so need to divide by 2
                if not np.allclose(adj, adj.T):
                    print(sample_id, 'not symmetric')
                degrees.extend(list(np.sum(adj, 1)))
                features.append(np.array(data['features'][sample_id]))

            # Create features over graphs as one-hot vectors for each node
            features_all = np.concatenate(features)
            features_min = features_all.min()
            num_features = int(features_all.max() - features_min + 1)  # number of possible values

            features_onehot = []
            for i, x in enumerate(features):
                feature_onehot = np.zeros((len(x), num_features))
                for node, value in enumerate(x):
                    feature_onehot[node, value - features_min] = 1
                if self.use_cont_node_attr:
                    feature_onehot = np.concatenate((feature_onehot, np.array(data['attr'][i])), axis=1)
                features_onehot.append(feature_onehot)

            if self.use_cont_node_attr:
                num_features = features_onehot[0].shape[1]

            shapes = [len(adj) for adj in data['adj_list']]
            labels = data['targets']  # graph class labels
            labels -= np.min(labels)  # to start from 0

            classes = np.unique(labels)
            num_classes = len(classes)

            if not np.all(np.diff(classes) == 1):
                print('making labels sequential, otherwise pytorch might crash')
                labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
                for lbl in range(num_classes):
                    labels_new[labels == classes[lbl]] = lbl
                labels = labels_new
                classes = np.unique(labels)
                assert len(np.unique(labels)) == num_classes, np.unique(labels)

            def stats(x):
                return (np.mean(x), np.std(x), np.min(x), np.max(x))

            print('N nodes avg/std/min/max: \t%.2f/%.2f/%d/%d' % stats(shapes))
            print('N edges avg/std/min/max: \t%.2f/%.2f/%d/%d' % stats(n_edges))
            print('Node degree avg/std/min/max: \t%.2f/%.2f/%d/%d' % stats(degrees))
            print('Node features dim: \t\t%d' % num_features)
            print('N classes: \t\t\t%d' % num_classes)
            print('Classes: \t\t\t%s' % str(classes))
            for lbl in classes:
                print('Class %d: \t\t\t%d samples' % (lbl, np.sum(labels == lbl)))

            for u in np.unique(features_all):
                print('feature {}, count {}/{}'.format(u, np.count_nonzero(features_all == u), len(features_all)))

            N_graphs = len(labels)  # number of samples (graphs) in data
            assert N_graphs == len(data['adj_list']) == len(features_onehot), 'invalid data'

            # Create test sets first
            train_ids, test_ids = split_ids(rnd_state.permutation(N_graphs), folds=folds)

            # Create train sets
            splits = []
            for fold in range(folds):
                splits.append({'train': train_ids[fold],
                               'test': test_ids[fold]})

            data['features_onehot'] = features_onehot
            data['targets'] = labels
            data['splits'] = splits
            data['N_nodes_max'] = np.max(shapes)  # max number of nodes
            data['num_features'] = num_features
            data['num_classes'] = num_classes

            self.data = data

        def parse_txt_file(self, fpath, line_parse_fn=None):
            with open(pjoin(self.data_dir, fpath), 'r') as f:
                lines = f.readlines()
            data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
            return data

        def read_graph_adj(self, fpath, nodes, graphs):
            edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
            adj_dict = {}
            for edge in edges:
                node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
                node2 = int(edge[1].strip()) - 1
                graph_id = nodes[node1]
                assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
                if graph_id not in adj_dict:
                    n = len(graphs[graph_id])
                    adj_dict[graph_id] = np.zeros((n, n))
                ind1 = np.where(graphs[graph_id] == node1)[0]
                ind2 = np.where(graphs[graph_id] == node2)[0]
                assert len(ind1) == len(ind2) == 1, (ind1, ind2)
                adj_dict[graph_id][ind1, ind2] = 1

            adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]

            return adj_list

        def read_graph_nodes_relations(self, fpath):
            graph_ids = self.parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
            nodes, graphs = {}, {}
            for node_id, graph_id in enumerate(graph_ids):
                if graph_id not in graphs:
                    graphs[graph_id] = []
                graphs[graph_id].append(node_id)
                nodes[node_id] = graph_id
            graph_ids = np.unique(list(graphs.keys()))
            for graph_id in graph_ids:
                graphs[graph_id] = np.array(graphs[graph_id])
            return nodes, graphs

        def read_node_features(self, fpath, nodes, graphs, fn):
            node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)
            node_features = {}
            for node_id, x in enumerate(node_features_all):
                graph_id = nodes[node_id]
                if graph_id not in node_features:
                    node_features[graph_id] = [None] * len(graphs[graph_id])
                ind = np.where(graphs[graph_id] == node_id)[0]
                assert len(ind) == 1, ind
                assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
                node_features[graph_id][ind[0]] = x
            node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
            return node_features_lst


# NN layers and models
class GraphConv(nn.Module):
    '''
    Graph Convolution Layer according to (T. Kipf and M. Welling, ICLR 2017)
    Additional tricks (power of adjacency matrix and weighted self connections) as in the Graph U-Net paper
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 n_relations=1,  # number of relation types (adjacency matrices)
                 activation=None,
                 adj_sq=False,
                 scale_identity=False):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features * n_relations, out_features=out_features)
        self.n_relations = n_relations
        self.activation = activation
        self.adj_sq = adj_sq
        self.scale_identity = scale_identity

    def laplacian_batch(self, A):
        batch, N = A.shape[:2]
        if self.adj_sq:
            A = torch.bmm(A, A)  # use A^2 to increase graph connectivity
        I = torch.eye(N).unsqueeze(0).to(args.device)
        if self.scale_identity:
            I = 2 * I  # increase weight of self connections
        A_hat = A + I
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, data):
        x, A, mask = data[:3]
        # print('in', x.shape, torch.sum(torch.abs(torch.sum(x, 2)) > 0))
        if len(A.shape) == 3:
            A = A.unsqueeze(3)
        x_hat = []
        for rel in range(self.n_relations):
            x_hat.append(torch.bmm(self.laplacian_batch(A[:, :, :, rel]), x))
        x = self.fc(torch.cat(x_hat, 2))

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(2)

        x = x * mask  # to make values of dummy nodes zeros again, otherwise the bias is added after applying self.fc which affects node embeddings in the following layers
        # print('out', x.shape, torch.sum(torch.abs(torch.sum(x, 2)) > 0))
        if self.activation is not None:
            x = self.activation(x)
        return (x, A, mask)


class GCN(nn.Module):
    '''
    Baseline Graph Convolutional Network with a stack of Graph Convolution Layers and global pooling over nodes.
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 filters=[64, 64, 64],
                 n_hidden=0,
                 dropout=0.2,
                 adj_sq=False,
                 scale_identity=False):
        super(GCN, self).__init__()

        # Graph convolution layers
        self.gconv = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                                out_features=f,
                                                activation=nn.ReLU(inplace=True),
                                                adj_sq=adj_sq,
                                                scale_identity=scale_identity) for layer, f in enumerate(filters)]))

        # Fully connected layers
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fc.append(nn.Linear(filters[-1], n_hidden))
            if dropout > 0:
                fc.append(nn.Dropout(p=dropout))
            n_last = n_hidden
        else:
            n_last = filters[-1]
        fc.append(nn.Linear(n_last, out_features))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        x = self.gconv(data)[0]
        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes (usually performs better than average)
        x = self.fc(x)
        return x


class GraphUnet(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 filters=[64, 64, 64],
                 n_hidden=0,
                 dropout=0.2,
                 adj_sq=False,
                 scale_identity=False,
                 shuffle_nodes=False,
                 visualize=False,
                 pooling_ratios=[0.8, 0.8]):
        super(GraphUnet, self).__init__()

        self.shuffle_nodes = shuffle_nodes
        self.visualize = visualize
        self.pooling_ratios = pooling_ratios
        # Graph convolution layers
        self.gconv = nn.ModuleList([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                              out_features=f,
                                              activation=nn.ReLU(inplace=True),
                                              adj_sq=adj_sq,
                                              scale_identity=scale_identity) for layer, f in enumerate(filters)])
        # Pooling layers
        self.proj = []
        for layer, f in enumerate(filters[:-1]):
            # Initialize projection vectors similar to weight/bias initialization in nn.Linear
            fan_in = filters[layer]
            p = Parameter(torch.Tensor(fan_in, 1))
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(p, -bound, bound)
            self.proj.append(p)
        self.proj = nn.ParameterList(self.proj)

        # Fully connected layers
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fc.append(nn.Linear(filters[-1], n_hidden))
            if dropout > 0:
                fc.append(nn.Dropout(p=dropout))
            n_last = n_hidden
        else:
            n_last = filters[-1]
        fc.append(nn.Linear(n_last, out_features))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        # data: [node_features, A, graph_support, N_nodes, label]
        if self.shuffle_nodes:
            # shuffle nodes to make sure that the model does not adapt to nodes order (happens in some cases)
            N = data[0].shape[1]
            idx = torch.randperm(N)
            data = (data[0][:, idx], data[1][:, idx, :][:, :, idx], data[2][:, idx], data[3])

        sample_id_vis, N_nodes_vis = -1, -1
        for layer, gconv in enumerate(self.gconv):
            N_nodes = data[3]

            # TODO: remove dummy or dropped nodes for speeding up forward/backward passes
            # data = (data[0][:, :N_nodes_max], data[1][:, :N_nodes_max, :N_nodes_max], data[2][:, :N_nodes_max], data[3])

            x, A = data[:2]

            B, N, _ = x.shape

            # visualize data
            if self.visualize and layer < len(self.gconv) - 1:
                for b in range(B):
                    if (layer == 0 and N_nodes[b] < 20 and N_nodes[b] > 10) or sample_id_vis > -1:
                        if sample_id_vis > -1 and sample_id_vis != b:
                            continue
                        if N_nodes_vis < 0:
                            N_nodes_vis = N_nodes[b]
                        plt.figure()
                        plt.imshow(A[b][:N_nodes_vis, :N_nodes_vis].data.cpu().numpy())
                        plt.title('layer %d, Input adjacency matrix' % (layer))
                        plt.savefig('input_adjacency_%d.png' % layer)
                        sample_id_vis = b
                        break

            mask = data[2].clone()  # clone as we are going to make inplace changes
            x = gconv(data)[0]  # graph convolution
            if layer < len(self.gconv) - 1:
                B, N, C = x.shape
                y = torch.mm(x.view(B * N, C), self.proj[layer]).view(B, N)  # project features
                y = y / (torch.sum(self.proj[layer] ** 2).view(1, 1) ** 0.5)  # node scores used for ranking below
                idx = torch.sort(y, dim=1)[1]  # get indices of y values in the ascending order                
                N_remove = (N_nodes.float() * (1 - self.pooling_ratios[layer])).long()  # number of removed nodes

                # sanity checks
                assert torch.all(
                    N_nodes > N_remove), 'the number of removed nodes must be large than the number of nodes'
                for b in range(B):
                    # check that mask corresponds to the actual (non-dummy) nodes
                    assert torch.sum(mask[b]) == float(N_nodes[b]), (torch.sum(mask[b]), N_nodes[b])

                N_nodes_prev = N_nodes
                N_nodes = N_nodes - N_remove

                for b in range(B):
                    idx_b = idx[b, mask[b, idx[b]] == 1]  # take indices of non-dummy nodes for current data example
                    assert len(idx_b) >= N_nodes[b], (
                        len(idx_b), N_nodes[b])  # number of indices must be at least as the number of nodes
                    mask[b, idx_b[:N_remove[b]]] = 0  # set mask values corresponding to the smallest y-values to 0

                # sanity checks
                for b in range(B):
                    # check that the new mask corresponds to the actual (non-dummy) nodes
                    assert torch.sum(mask[b]) == float(N_nodes[b]), (
                        b, torch.sum(mask[b]), N_nodes[b], N_remove[b], N_nodes_prev[b])
                    # make sure that y-values of selected nodes are larger than of dropped nodes
                    s = torch.sum(y[b] >= torch.min((y * mask.float())[b]))
                    assert s >= float(N_nodes[b]), (s, N_nodes[b], (y * mask.float())[b])

                mask = mask.unsqueeze(2)
                x = x * torch.tanh(y).unsqueeze(2) * mask  # propagate only part of nodes using the mask
                A = mask * A * mask.view(B, 1, N)
                mask = mask.squeeze()
                data = (x, A, mask, N_nodes)

                # visualize data
                if self.visualize and sample_id_vis > -1:
                    b = sample_id_vis
                    plt.figure()
                    plt.imshow(y[b].view(N, 1).expand(N, 2)[:N_nodes_vis].data.cpu().numpy())
                    plt.title('Node ranking')
                    plt.colorbar()
                    plt.savefig('nodes_ranking_%d.png' % layer)
                    plt.figure()
                    plt.imshow(mask[b].view(N, 1).expand(N, 2)[:N_nodes_vis].data.cpu().numpy())
                    plt.title('Pooled nodes (%d/%d)' % (mask[b].sum(), N_nodes_prev[b]))
                    plt.savefig('pooled_nodes_mask_%d.png' % layer)
                    plt.figure()
                    plt.imshow(A[b][:N_nodes_vis, :N_nodes_vis].data.cpu().numpy())
                    plt.title('Pooled adjacency matrix')
                    plt.savefig('pooled_adjacency_%d.png' % layer)
                    print('layer %d: visualizations saved ' % layer)

        if self.visualize and sample_id_vis > -1:
            self.visualize = False  # to prevent visualization for the following batches

        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes
        x = self.fc(x)
        return x


class MGCN(nn.Module):
    '''
    Multigraph Convolutional Network based on (B. Knyazev et al., "Spectral Multigraph Networks for Discovering and Fusing Relationships in Molecules")
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 n_relations,
                 filters=[64, 64, 64],
                 n_hidden=0,
                 dropout=0.2,
                 adj_sq=False,
                 scale_identity=False):
        super(MGCN, self).__init__()

        # Graph convolution layers
        self.gconv = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                                out_features=f,
                                                n_relations=n_relations,
                                                activation=nn.ReLU(inplace=True),
                                                adj_sq=adj_sq,
                                                scale_identity=scale_identity) for layer, f in enumerate(filters)]))

        # Edge prediction NN
        self.edge_pred = nn.Sequential(nn.Linear(in_features * 2, 32),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(32, 1))

        # Fully connected layers
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fc.append(nn.Linear(filters[-1], n_hidden))
            if dropout > 0:
                fc.append(nn.Dropout(p=dropout))
            n_last = n_hidden
        else:
            n_last = filters[-1]
        fc.append(nn.Linear(n_last, out_features))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        # data: [node_features, A, graph_support, N_nodes, label]

        # Predict edges based on features
        x = data[0]
        B, N, C = x.shape
        mask = data[2]
        # find indices of nodes
        x_cat, idx = [], []
        for b in range(B):
            n = int(mask[b].sum())
            node_i = torch.nonzero(mask[b]).repeat(1, n).view(-1, 1)
            node_j = torch.nonzero(mask[b]).repeat(n, 1).view(-1, 1)
            triu = (node_i < node_j).squeeze()  # skip loops and symmetric connections
            x_cat.append(torch.cat((x[b, node_i[triu]], x[b, node_j[triu]]), 2).view(int(torch.sum(triu)), C * 2))
            idx.append((node_i * N + node_j)[triu].squeeze())

        x_cat = torch.cat(x_cat)
        idx_flip = np.concatenate((np.arange(C, 2 * C), np.arange(C)))
        # predict values and encourage invariance to nodes order
        y = torch.exp(0.5 * (self.edge_pred(x_cat) + self.edge_pred(x_cat[:, idx_flip])).squeeze())
        A_pred = torch.zeros(B, N * N, device=args.device)
        c = 0
        for b in range(B):
            A_pred[b, idx[b]] = y[c:c + idx[b].nelement()]
            c += idx[b].nelement()
        A_pred = A_pred.view(B, N, N)
        A_pred = (A_pred + A_pred.permute(0, 2, 1))  # assume undirected edges

        # Use both annotated and predicted adjacency matrices to learn a GCN
        data = (x, torch.stack((data[1], A_pred), 3), mask)
        x = self.gconv(data)[0]
        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes
        x = self.fc(x)
        return x


def collate_batch(batch):
    '''
    Creates a batch of same size graphs by zero-padding node features and adjacency matrices up to
    the maximum number of nodes in the CURRENT batch rather than in the entire dataset.
    Graphs in the batches are usually much smaller than the largest graph in the dataset, so this method is fast.
    :param batch: batch in the PyTorch Geometric format or [node_features*batch_size, A*batch_size, label*batch_size]
    :return: [node_features, A, graph_support, N_nodes, label]
    '''
    B = len(batch)
    if args.torch_geom:
        N_nodes = [len(batch[b].x) for b in range(B)]
        C = batch[0].x.shape[1]
    else:
        N_nodes = [len(batch[b][1]) for b in range(B)]
        C = batch[0][0].shape[1]
    N_nodes_max = int(np.max(N_nodes))

    graph_support = torch.zeros(B, N_nodes_max)
    A = torch.zeros(B, N_nodes_max, N_nodes_max)
    x = torch.zeros(B, N_nodes_max, C)
    for b in range(B):
        if args.torch_geom:
            x[b, :N_nodes[b]] = batch[b].x
            A[b].index_put_((batch[b].edge_index[0], batch[b].edge_index[1]), torch.Tensor([1]))
        else:
            x[b, :N_nodes[b]] = batch[b][0]
            A[b, :N_nodes[b], :N_nodes[b]] = batch[b][1]
        graph_support[b][:N_nodes[b]] = 1  # mask with values of 0 for dummy (zero padded) nodes, otherwise 1

    N_nodes = torch.from_numpy(np.array(N_nodes)).long()
    labels = torch.from_numpy(np.array([batch[b].y if args.torch_geom else batch[b][2] for b in range(B)])).long()
    return [x, A, graph_support, N_nodes, labels]


print('Loading data')

if args.torch_geom:
    dataset = TUDataset('./data/%s/' % args.dataset, name=args.dataset,
                        use_node_attr=args.use_cont_node_attr)
    train_ids, test_ids = split_ids(rnd_state.permutation(len(dataset)), folds=n_folds)

else:
    datareader = DataReader(data_dir='./data/%s/' % args.dataset,
                            rnd_state=rnd_state,
                            folds=n_folds,
                            use_cont_node_attr=args.use_cont_node_attr)

acc_folds = []
for fold_id in range(n_folds):

    loaders = []
    for split in ['train', 'test']:
        if args.torch_geom:
            gdata = dataset[torch.from_numpy((train_ids if split.find('train') >= 0 else test_ids)[fold_id])]
        else:
            gdata = GraphData(fold_id=fold_id,
                              datareader=datareader,
                              split=split)

        loader = DataLoader(gdata,
                            batch_size=args.batch_size,
                            shuffle=split.find('train') >= 0,
                            num_workers=args.threads,
                            collate_fn=collate_batch)
        loaders.append(loader)

    print('\nFOLD {}, train {}, test {}'.format(fold_id, len(loaders[0].dataset), len(loaders[1].dataset)))

    if args.model == 'gcn':
        model = GCN(in_features=loaders[0].dataset.num_features,
                    out_features=loaders[0].dataset.num_classes,
                    n_hidden=args.n_hidden,
                    filters=args.filters,
                    dropout=args.dropout,
                    adj_sq=args.adj_sq,
                    scale_identity=args.scale_identity).to(args.device)
    elif args.model == 'unet':
        model = GraphUnet(in_features=loaders[0].dataset.num_features,
                          out_features=loaders[0].dataset.num_classes,
                          n_hidden=args.n_hidden,
                          filters=args.filters,
                          dropout=args.dropout,
                          adj_sq=args.adj_sq,
                          scale_identity=args.scale_identity,
                          shuffle_nodes=args.shuffle_nodes,
                          visualize=args.visualize).to(args.device)
    elif args.model == 'mgcn':
        model = MGCN(in_features=loaders[0].dataset.num_features,
                     out_features=loaders[0].dataset.num_classes,
                     n_relations=2,
                     n_hidden=args.n_hidden,
                     filters=args.filters,
                     dropout=args.dropout,
                     adj_sq=args.adj_sq,
                     scale_identity=args.scale_identity).to(args.device)

    else:
        raise NotImplementedError(args.model)

    print('\nInitialize model')
    print(model)
    train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print('N trainable parameters:', np.sum([p.numel() for p in train_params]))

    optimizer = optim.Adam(train_params, lr=args.lr, weight_decay=args.wd, betas=(0.5, 0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)


    # Normalization of continuous node features
    # if args.use_cont_node_attr:
    #     x = []
    #     for batch_idx, data in enumerate(loaders[0]):
    #         if args.torch_geom:
    #             node_attr_dim = loaders[0].dataset.props['node_attr_dim']
    #         x.append(data[0][:, :, :node_attr_dim].view(-1, node_attr_dim).data)
    #     x = torch.cat(x)
    #     mn, sd = torch.mean(x, dim=0).to(args.device), torch.std(x, dim=0).to(args.device) + 1e-5
    #     print(mn, sd)
    # else:
    #     mn, sd = 0, 1

    # def norm_features(x):
    #     x[:, :, :node_attr_dim] = (x[:, :, :node_attr_dim] - mn) / sd

    def train(train_loader):
        scheduler.step()
        model.train()
        start = time.time()
        train_loss, n_samples = 0, 0
        for batch_idx, data in enumerate(train_loader):
            for i in range(len(data)):
                data[i] = data[i].to(args.device)
            # if args.use_cont_node_attr:
            #     data[0] = norm_features(data[0])
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, data[4])
            loss.backward()
            optimizer.step()
            time_iter = time.time() - start
            train_loss += loss.item() * len(output)
            n_samples += len(output)
            if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader) - 1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
                    epoch + 1, n_samples, len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples,
                    time_iter / (batch_idx + 1)))


    def test(test_loader):
        model.eval()
        start = time.time()
        test_loss, correct, n_samples = 0, 0, 0
        for batch_idx, data in enumerate(test_loader):
            for i in range(len(data)):
                data[i] = data[i].to(args.device)
            # if args.use_cont_node_attr:
            #     data[0] = norm_features(data[0])
            output = model(data)
            loss = loss_fn(output, data[4], reduction='sum')
            test_loss += loss.item()
            n_samples += len(output)
            pred = output.detach().cpu().max(1, keepdim=True)[1]

            correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()

        acc = 100. * correct / n_samples
        print('Test set (epoch {}): Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) \tsec/iter: {:.4f}\n'.format(
            epoch + 1,
            test_loss / n_samples,
            correct,
            n_samples,
            acc, (time.time() - start) / len(test_loader)))
        return acc

    loss_fn = F.cross_entropy
    for epoch in range(args.epochs):
        train(loaders[0])
        acc = test(loaders[1])
    acc_folds.append(acc)

print(acc_folds)
print('{}-fold cross validation avg acc (+- std): {} ({})'.format(n_folds, np.mean(acc_folds), np.std(acc_folds)))
