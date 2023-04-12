# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead
from mmdet.models.utils import build_linear_layer
from mmcv.cnn import ConvModule
import torch
from mmdet.models.losses import accuracy
from mmdet.models.builder import HEADS, build_loss
import itertools


@HEADS.register_module()
class ConvFCDisentangledBBoxHead(ConvFCBBoxHead):
    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 class_heads=2,
                 weight_similarity_loss=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        if weight_similarity_loss is not None:
            self.weight_similarity_loss = build_loss(weight_similarity_loss)
            self.cos_similarity  = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        else:
            self.weight_similarity_loss = None

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        neck = [(self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)) for i in range(class_heads)]
        self.cls_last_dim = neck[0][-1]

        # add cls specific branch
        # self.cls_convs1, self.cls_fcs1, self.cls_last_dim = \
        #     self._add_conv_fc_branch(
        #         self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        _, self.hidden_switch, self.switch_dim = self._add_conv_fc_branch(0, 1, self.shared_out_channels)


        self.cls_convs = nn.ModuleList([net[0].cuda() for net in neck])
        self.cls_fcs = nn.ModuleList([net[1].cuda() for net in neck])


        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            # self.fc_cls = build_linear_layer(
            #     self.cls_predictor_cfg,
            #     in_features=self.cls_last_dim,
            #     out_features=cls_channels)

            self.fc_cls = nn.ModuleList([build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels).cuda() for k in range(class_heads)])

            if class_heads>2:
                self.switch = build_linear_layer(
                    self.cls_predictor_cfg,
                    in_features=self.switch_dim,
                    out_features=class_heads)
            else:
                self.switch = build_linear_layer(
                    self.cls_predictor_cfg,
                    in_features=self.switch_dim,
                    out_features=1)

        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_switch = x
        x_reg = x
        cls_score = []
        for convnet,ann,classifier in zip(self.cls_convs,self.cls_fcs,self.fc_cls):
            x_cls = x
            for conv in convnet:
                x_cls = conv(x_cls)
            if x_cls.dim() > 2:
                if self.with_avg_pool:
                    x_cls = self.avg_pool(x_cls)
                x_cls = x_cls.flatten(1)
        
            for fc in ann:
                x_cls = self.relu(fc(x_cls))

            if self.with_cls:
                cls_score.append(classifier(x_cls))
            else:
                cls_score = None
                
        if self.with_cls:
            cls_score= torch.cat(cls_score,1)
        else:
            cls_score = None


        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        if x_switch.dim() > 2:
            if self.with_avg_pool:
                x_switch = self.avg_pool(x_switch)
            x_switch = x_switch.flatten(1)
        
        
        switch_logit = self.switch(self.relu(self.hidden_switch[0](x_switch))) if self.with_cls else None
        switch_logit = torch.clamp(switch_logit,min=-12,max=12)
        cls_score = torch.cat([cls_score,switch_logit],1) # channel-wise concatenation

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        return cls_score, bbox_pred


    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        # weight norm similarity loss
        if self.weight_similarity_loss is not None:
            fc_cls_weights=[next(net.parameters()) for net in self.fc_cls]
            fc_cls_weight_sim = torch.cat([torch.abs(self.cos_similarity(pair[0],pair[1])) for pair in list(itertools.combinations(fc_cls_weights, 2))],dim=0)
            fc_cls_weight_sim = torch.clamp(fc_cls_weight_sim,min=0.0001,max=0.9999)
            weight_sim_loss_ = self.weight_similarity_loss(fc_cls_weight_sim,torch.zeros_like(fc_cls_weight_sim))
            # weight_sim_loss_ = self.weight_similarity_loss(fc_cls_weight_sim,torch.ones_like(fc_cls_weight_sim))
            losses['weight_similarity_loss'] = weight_sim_loss_

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses




@HEADS.register_module()
class DisentangledMultiActivationBBoxHead(ConvFCDisentangledBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(DisentangledMultiActivationBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=0,
            num_cls_convs=0,
            num_cls_fcs=2,
            num_reg_convs=4,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        


        
    
        

    
    