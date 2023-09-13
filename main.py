import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
from DGCNN_embedding import DGCNN
from mlp_dropout import MLPClassifier, MLPRegression
from sklearn import metrics
from util import cmd_args, load_data

class Classifier(nn.Module):
    def __init__(self, regression=False):
        super(Classifier, self).__init__()
        self.regression = regression
        if cmd_args.gm == 'DGCNN':
            model = DGCNN
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()

        if cmd_args.gm == 'DGCNN':
            self.gnn = model(latent_dim=cmd_args.latent_dim,
                            output_dim=cmd_args.out_dim,
                            num_node_feats=cmd_args.feat_dim+cmd_args.attr_dim,
                            num_edge_feats=cmd_args.edge_feat_dim,
                            k=cmd_args.sortpooling_k, 
                            conv1d_activation=cmd_args.conv1d_activation)
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            if cmd_args.gm == 'DGCNN':
                out_dim = self.gnn.dense_dim
            else:
                out_dim = cmd_args.latent_dim
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.hidden, num_class=cmd_args.num_class, with_dropout=cmd_args.dropout)
        if regression:
            self.mlp = MLPRegression(input_size=out_dim, hidden_size=cmd_args.hidden, with_dropout=cmd_args.dropout)

    def PrepareFeatureLabel(self, batch_graph):
        if self.regression:
            labels = torch.FloatTensor(len(batch_graph))
        else:
            labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        if cmd_args.edge_feat_dim > 0:
            edge_feat_flag = True
            concat_edge_feat = []
        else:
            edge_feat_flag = False

        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag == True:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag == True:
                tmp = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
                concat_feat.append(tmp)
            if edge_feat_flag == True:
                if batch_graph[i].edge_features is not None:  # in case no edge in graph[i]
                    tmp = torch.from_numpy(batch_graph[i].edge_features).type('torch.FloatTensor')
                    concat_edge_feat.append(tmp)

        if node_tag_flag == True:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, cmd_args.feat_dim)
            node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag == True:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features
        
        if edge_feat_flag == True:
            edge_feat = torch.cat(concat_edge_feat, 0)

        if cmd_args.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()
            if edge_feat_flag == True:
                edge_feat = edge_feat.cuda()

        if edge_feat_flag == True:
            return node_feat, edge_feat, labels
        return node_feat, labels

    def forward(self, batch_graph):
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat)
        return self.mlp(embed, labels, [graph.nodegroup for graph in batch_graph])

    def output_features(self, batch_graph):
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat)
        return embed, labels
        

def loop_dataset(g_list, classifier, sample_idxes, optimizer=None, bsize=cmd_args.batch_size):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_scores = []

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets
        if classifier.regression:
            pred, mae, loss = classifier(batch_graph)
            all_scores.append(pred.cpu().detach())  # for binary classification
        else:
            logits, loss, acc = classifier(batch_graph)
            all_scores.append(logits[:, 1].cpu().detach())  # for binary classification

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().detach().numpy()
        if classifier.regression:
            pbar.set_description('MSE_loss: %0.5f MAE_loss: %0.5f' % (loss, mae) )
            total_loss.append( np.array([loss, mae]) * len(selected_idx))
        else:
            pbar.set_description('loss: %0.5f head_acc: %0.5f med_acc: %0.5f tail_acc: %0.5f' % (loss, acc[0], acc[1], acc[2]))
            total_loss.append(np.array([loss] + [sum(acc) / 3] + acc) * len(selected_idx))


        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples

    # np.savetxt('test_scores.txt', all_scores)  # output test predictions
    
    # if not classifier.regression and cmd_args.printAUC:
    #     all_targets = np.array(all_targets)
    #     fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
    #     auc = metrics.auc(fpr, tpr)
    #     avg_loss = np.concatenate((avg_loss, [auc]))
    # else:
    #     avg_loss = np.concatenate((avg_loss, [0.0]))
    
    return avg_loss


if __name__ == '__main__':
    print(cmd_args)
    SEEDS = [0, 1, 2, 3, 4]

    test_record = torch.zeros(len(SEEDS))
    valid_record = torch.zeros(len(SEEDS))
    tail_record = torch.zeros(len(SEEDS))
    medium_record = torch.zeros(len(SEEDS))
    head_record = torch.zeros(len(SEEDS))

    for seed in SEEDS:
        print(f"Training with SEED - {seed}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_graphs, val_graphs, test_graphs = load_data()
        print("Number of Train Graphs: ", len(train_graphs))
        print("Number of Val Graphs: ", len(val_graphs))
        print("Number of Test Graphs: ", len(test_graphs))

        best_acc = 0

        if cmd_args.sortpooling_k <= 1:
            num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
            cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
            cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
            print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

        classifier = Classifier()
        if cmd_args.mode == 'gpu':
            classifier = classifier.cuda()

        optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

        train_idxes = list(range(len(train_graphs)))
        best_loss = 1
        for epoch in range(cmd_args.num_epochs):
            random.shuffle(train_idxes)
            classifier.train()
            avg_loss = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer)
            # if not cmd_args.printAUC:
            #     avg_loss[2] = 0.0
            print('\033[92maverage training of epoch %d: loss %.5f acc %.5f head_acc %.5f med_acc %.5f tail_acc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3], avg_loss[4]))

            classifier.eval()
            val_loss = loop_dataset(val_graphs, classifier, list(range(len(val_graphs))))
            # if not cmd_args.printAUC:
            #     test_loss[2] = 0.0
            print('\033[93maverage val of epoch %d: loss %.5f acc %.5f head_acc %.5f med_acc %.5f tail_acc %.5f\033[0m' % (epoch, val_loss[0], val_loss[1], val_loss[2], val_loss[3], val_loss[4]))

            if best_loss > val_loss[0] and best_acc < val_loss[1]:
                classifier.eval()
                test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
                print(
                    '\033[91maverage test of epoch %d: loss %.5f acc %.5f head_acc %.5f med_acc %.5f tail_acc %.5f\033[0m' % (
                    epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4]))

                best_loss = val_loss[0]
                best_acc = val_loss[1]
                best_test_metrics = test_loss
                best_val_metrics = val_loss

        with open(cmd_args.data + '_acc_results.txt', 'a+') as f:
            f.write(str(test_loss[1]) + '\n')

        if cmd_args.printAUC:
            with open(cmd_args.data + '_auc_results.txt', 'a+') as f:
                f.write(str(test_loss[2]) + '\n')

        if cmd_args.extract_features:
            features, labels = classifier.output_features(train_graphs)
            labels = labels.type('torch.FloatTensor')
            np.savetxt('extracted_features_train.txt', torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')
            features, labels = classifier.output_features(test_graphs)
            labels = labels.type('torch.FloatTensor')
            np.savetxt('extracted_features_test.txt', torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')

        test_record[seed] = best_test_metrics[1]
        valid_record[seed] = best_val_metrics[1]
        head_record[seed] = best_test_metrics[2]
        medium_record[seed] = best_test_metrics[3]
        tail_record[seed] = best_test_metrics[4]

    args = cmd_args
    with open("metrics.txt", "a") as txt_file:
        txt_file.write(f"Dataset: {args.data}, \n"
                       # f"Alpha: {args.alpha}, \n"
                       # f"Mu: {args.mu1}, \n"
                       f"Valid Mean: {round(valid_record.mean().item(), 4)} \n"
                       f"Std Valid Mean: {round(valid_record.std().item(), 4)} \n"
                       f"Test Mean: {round(test_record.mean().item(), 4)} \n"
                       f"Std Test Mean: {round(test_record.std().item(), 4)} \n"
                       f"Head Mean: {round(head_record.mean().item(), 4)} \n"
                       f"Std Head Mean: {round(head_record.std().item(), 4)} \n"
                       f"Medium Mean: {round(medium_record.mean().item(), 4)} \n"
                       f"Std Medium Mean: {round(medium_record.std().item(), 4)}, \n"
                       f"Tail Mean: {round(tail_record.mean().item(), 4)} \n"
                       f"Std Tail Mean: {round(tail_record.std().item(), 4)} \n\n"
                       )

    # os.makedirs("sortpool", exist_ok=True)
    # os.makedirs(os.path.join("sortpool", args.dataset), exist_ok=True)
    #
    # with open(os.path.join("sortpool", args.dataset, f"{args.dataset}_{args.alpha}_{args.mu1}.json", "w")) as json_file:
    #     json.dump({"Dataset": args.dataset,
    #                # "Alpha": {args.alpha},
    #                # "Mu": args.mu1,
    #                "Valid Mean": round(valid_record.mean().item(), 4),
    #                "Std Valid Mean": round(valid_record.std().item(), 4),
    #                "Test Mean": round(test_record.mean().item(), 4),
    #                "Std Test Mean": round(test_record.std().item(), 4),
    #                "Head Mean": round(head_record.mean().item(), 4),
    #                "Std Head Mean": round(head_record.std().item(), 4),
    #                "Medium Mean": round(medium_record.mean().item(), 4),
    #                "Std Medium Mean": round(medium_record.std().item(), 4),
    #                "Tail Mean": round(tail_record.mean().item(), 4),
    #                "Std Tail Mean": round(tail_record.std().item(), 4)}, json_file
    #               )

    # print(seed_test_metrics.mean(axis=0))
    # print(seed_val_metrics.mean(axis=0))
    # print(seed_test_metrics.std(axis=0))
    # print(seed_val_metrics.std(axis=0))


