from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import torch.optim as optim
from tqdm import tqdm
import pdb

sys.path.append('%s/lib' % os.path.dirname(os.path.realpath(__file__)))
from pytorch_util import weights_init

class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size, with_dropout=False):
        super(MLPRegression, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)
        self.with_dropout = with_dropout

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)

        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)
        pred = self.h2_weights(h1)[:, 0]

        if y is not None:
            y = Variable(y)
            mse = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            mae = mae.cpu().detach()
            return pred, mae, mse
        else:
            return pred

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, with_dropout=False):
        super(MLPClassifier, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.with_dropout = with_dropout

        weights_init(self)

    def forward(self, x, y = None, groups = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)

        logits = self.h2_weights(h1)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)

            pred = logits.data.max(1, keepdim=True)[1]

            head_preds = []
            head_y = []
            med_preds = []
            med_y = []
            tail_preds = []
            tail_y = []

            for idx, id in enumerate(groups):
                if id == 2:
                    head_preds.append(pred[idx].cpu().item())
                    head_y.append(y.data[idx].cpu().item())
                elif id == 1:
                    med_preds.append(pred[idx].cpu().item())
                    med_y.append(y.data[idx].cpu().item())
                elif id == 0:
                    tail_preds.append(pred[idx].cpu().item())
                    tail_y.append(y.data[idx].cpu().item())
                else:
                    assert False

            # print(f"Head ACC: {accuracy_score(head_y, head_preds)}, Medium ACC: {accuracy_score(head_y, head_preds)}, "
            #       f"Tail ACC: {accuracy_score(tail_y, tail_preds)}")
            # acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            return logits, loss, [accuracy_score(head_y, head_preds), accuracy_score(head_y, head_preds),
                                  accuracy_score(tail_y, tail_preds)]
        else:
            return logits
