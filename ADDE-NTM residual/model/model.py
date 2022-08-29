# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np
import torch
import torch.nn as nn
import torchvision
import ntm_manip
import torch.nn.functional as F
from src.loss_function import hash_labels, TripletSemiHardLoss


class NTM_extractor(nn.Module):
    """
    Extract attribute-specific embeddings and add attribute predictor for each.
    Args:
        attr_nums: 1-D list of numbers of attribute values for each attribute
        backbone: String that indicate the name of pretrained backbone
        dim_chunk: int, the size of each attribute-specific embedding
    """
    def __init__(self, model_feature, model_ntm):
        super(NTM_extractor, self).__init__()

        self.model_feature = model_feature
        self.model_ntm = model_ntm

    def forward(self, indicator, imgs, labels, train_loader, optimizer, args, train):
        """
        Returns:
            dis_feat: a list of extracted attribute-specific embeddings
            attr_classification_out: a list of classification prediction results for each attribute
        """
        feats = {}
        cls_outs = {}
        if train:
            for key in imgs.keys():
                feats[key], cls_outs[key] = self.model_feature(imgs[key])
        else:
            dis_feat, _ = self.model_feature(imgs)

        # residual_feat = memory(indicator)
        # input_memory = ntm_manip.memory_state_tensor(self)
        input_memory = None
        residual_feat = ntm_manip.manipulate_ntm(self, optimizer, indicator, train, input_memory)

        if train:
            feat_manip = torch.cat(feats['ref'], 1) + residual_feat
            feat_manip_split = list(torch.split(feat_manip, args.dim_chunk, dim=1))

            cls_outs_manip = []
            for attr_id, layer in enumerate(self.model_feature.attr_classifier):
                cls_outs_manip.append(layer(feat_manip_split[attr_id]).squeeze())

        else:
            feat_manip = torch.cat(dis_feat, 1) + residual_feat
            total_loss = -1
            feats = []
            cls_outs = []
            cls_outs_manip = []
        return feat_manip, input_memory, feats, cls_outs, cls_outs_manip

