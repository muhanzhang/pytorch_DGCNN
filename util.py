from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os
#import cPickle as cp
#import _pickle as cp  # python3 compatability
import networkx as nx
import pdb
import argparse
import torch

cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-gm', default='DGCNN', help='gnn model to use')
cmd_opt.add_argument('-data', default=None, help='data folder name')
cmd_opt.add_argument('-batch_size', type=int, default=50, help='minibatch size')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-feat_dim', type=int, default=0, help='dimension of discrete node feature (maximum node tag)')
cmd_opt.add_argument('-edge_feat_dim', type=int, default=0, help='dimension of edge features')
cmd_opt.add_argument('-num_class', type=int, default=0, help='#classes')
cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')
cmd_opt.add_argument('-test_number', type=int, default=0, help='if specified, will overwrite -fold and use the last -test_number graphs as testing data')
cmd_opt.add_argument('-num_epochs', type=int, default=1000, help='number of epochs')
cmd_opt.add_argument('-latent_dim', type=str, default='64', help='dimension(s) of latent layers')
cmd_opt.add_argument('-sortpooling_k', type=float, default=30, help='number of nodes kept after SortPooling')
cmd_opt.add_argument('-conv1d_activation', type=str, default='ReLU', help='which nn activation layer to use')
cmd_opt.add_argument('-out_dim', type=int, default=1024, help='graph embedding output size')
cmd_opt.add_argument('-hidden', type=int, default=100, help='dimension of mlp hidden layer')
cmd_opt.add_argument('-max_lv', type=int, default=4, help='max rounds of message passing')
cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
cmd_opt.add_argument('-dropout', type=bool, default=False, help='whether add dropout after dense layer')
cmd_opt.add_argument('-printAUC', type=bool, default=False, help='whether to print AUC (for binary classification only)')
cmd_opt.add_argument('-extract_features', type=bool, default=False, help='whether to extract final graph features')

cmd_args, _ = cmd_opt.parse_known_args()

cmd_args.latent_dim = [int(x) for x in cmd_args.latent_dim.split('-')]
if len(cmd_args.latent_dim) == 1:
    cmd_args.latent_dim = cmd_args.latent_dim[0]

class GNNGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())
        self.nodegroup = None

        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)        
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])
        
        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):  
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert(type(edge_features.values()[0]) == np.ndarray) 
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in edge_features.items()}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)


def load_data():

    if cmd_args.data == "PROTEINS":
        K = [0, 371, 742, 1113]
    elif cmd_args.data == "PTC":
        K = [0, 115, 230, 344]
    elif cmd_args.data == "IMDBBINARY":
        K = [0, 333, 666, 1000]
    elif cmd_args.data == "DD":
        K = [0, 393, 785, 1178]
    elif cmd_args.data == "FRANK":
        K =[0, 1445, 2890, 4337]

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('data/%s/%s.txt' % (cmd_args.data, cmd_args.data), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if attr is not None:
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            #assert len(g.edges()) * 2 == n_edges  (some graphs in COLLAB have self-loops, ignored here)
            assert len(g) == n
            g_list.append(GNNGraph(g, l, node_tags, node_features))
    nodes = list()
    for g in g_list:
        g.label = label_dict[g.label]

    nodes = torch.zeros(len(g_list))
    for i in range(len(g_list)):
        nodes[i] = g_list[i].num_nodes

    _, ind = torch.sort(nodes, descending=True)

    print(ind)

    head = 0
    med = 0
    tail = 0

    for i in ind[K[0]:K[1]]:
        head += 1
        g_list[i].nodegroup = 2
    for i in ind[K[1]:K[2]]:
        med += 1
        g_list[i].nodegroup = 1
    for i in ind[K[2]:K[3]]:
        tail += 1
        g_list[i].nodegroup = 0

    cmd_args.num_class = len(label_dict)
    cmd_args.feat_dim = len(feat_dict) # maximum node label (tag)
    cmd_args.edge_feat_dim = 0
    if node_feature_flag == True:
        cmd_args.attr_dim = node_features.shape[1] # dim of node features (attributes)
    else:
        cmd_args.attr_dim = 0

    print('# classes: %d' % cmd_args.num_class)
    print('# maximum node tag: %d' % cmd_args.feat_dim)
    print('# Number of Head Graphs: ', head)
    print('# Number of Medium Graphs: ', med)
    print('# Number of Tail Graphs: ', tail)

    assert n_g == head + tail + med

    random.shuffle(g_list)

    train_graphs = int(0.7 * n_g)
    val_graphs = int(0.1 * n_g)

    # if cmd_args.test_number == 0:
    #     train_idxes = np.loadtxt('data/%s/10fold_idx/train_idx-%d.txt' % (cmd_args.data, cmd_args.fold), dtype=np.int32).tolist()
    #     test_idxes = np.loadtxt('data/%s/10fold_idx/test_idx-%d.txt' % (cmd_args.data, cmd_args.fold), dtype=np.int32).tolist()
    #     return [g_list[i] for i in train_idxes], [g_list[i] for i in test_idxes]
    # else:
        # return g_list[: n_g - cmd_args.test_number], g_list[n_g - cmd_args.test_number :]
    return g_list[:train_graphs], g_list[train_graphs: train_graphs + val_graphs], g_list[train_graphs + val_graphs:]



