import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from torch.nn.modules import Module
import numpy as np

def build_loss(cfg: Dict) -> Module:

    loss_type = cfg['loss_type']

    if loss_type == 'l1_loss':
        return nn.L1Loss()
    elif loss_type == 'mse_loss':
        return nn.MSELoss()
    elif loss_type == 'cross_entropy_loss':
        return nn.CrossEntropyLoss()
    elif loss_type == 'nll_loss':
        return nn.NLLLoss()
    elif loss_type == 'bce_loss':
        return nn.BCELoss()
    elif loss_type == 'bce_with_logits_loss':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'triplet_margin_loss':
        return nn.TripletMarginLoss()
    elif loss_type == 'joint_edge_seg_loss':
        return JointEdgeSegLoss(cfg['num_classes'])
    else:
        raise Exception('invalid loss function, add the custom loss function to file')
 
class JointEdgeSegLoss(nn.Module):
    def __init__(self, classes, edge_weight=1, seg_weight=1):
        super(JointEdgeSegLoss, self).__init__()

        self.num_classes = classes
        self.seg_loss = nn.CrossEntropyLoss()

        self.edge_weight = edge_weight
        self.seg_weight = seg_weight

    def bce2d(self, input, target):
        # n, c, h, w = input.size()
    
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t ==1)
        neg_index = (target_t ==0)
        ignore_index=(target_t >1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index=ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num*1.0 / sum_num
        weight[neg_index] = pos_num*1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')
        return loss

    def forward(self, inputs, targets):
        segin, edgein = inputs
        segmask, edgemask = targets

        main_loss = 0

        main_loss += self.seg_weight * self.seg_loss(segin, segmask) # seg_loss
        main_loss += self.edge_weight * 20 * self.bce2d(edgein, edgemask) # edge_loss

        loss = main_loss.mean()   

        return loss
