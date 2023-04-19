# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from .bbox_head import BBoxHead
import torch


@HEADS.register_module()
class MultiSibling(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

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
                 *args,
                 **kwargs):
        super(MultiSibling, self).__init__(
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
        self.class_heads = class_heads

        # add shared convs and fcs
        shared_neck = [(self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)) for h in range(self.class_heads)]
        
        self.shared_out_channels = shared_neck[0][-1]
        self.shared_convs = nn.ModuleList([net[0].cuda() for net in shared_neck])
        self.shared_fcs = nn.ModuleList([net[1].cuda() for net in shared_neck])
        
        

        # add cls specific branch
        cls_neck = [(self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)) for h in range(self.class_heads)]
        self.cls_last_dim = cls_neck[0][-1]
        self.cls_convs = nn.ModuleList([net[0].cuda() for net in cls_neck])
        self.cls_fcs = nn.ModuleList([net[1].cuda() for net in cls_neck])

        # add reg specific branch
        reg_neck = [(self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)) for h in range(self.class_heads)]
        self.reg_last_dim = reg_neck[0][-1]
        self.reg_convs = nn.ModuleList([net[0].cuda() for net in cls_neck])
        self.reg_fcs = nn.ModuleList([net[1].cuda() for net in cls_neck])
        

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
            self.fc_cls = nn.ModuleList([build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels) for h in range(self.class_heads)])
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.ModuleList([build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg) for h in range(self.class_heads)])
            
        _, self.hidden_switch, self.switch_dim = self._add_conv_fc_branch(0, 1, self.in_channels,True)
        
        self.switch = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.switch_dim,
                out_features=1)

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
        x_out = [x for h in range(self.class_heads)]
        x_switch = x
        if self.num_shared_convs > 0:
            for h in range(self.class_heads):
                for conv in self.shared_convs[h]:
                    x_out[h] = conv(x_out[h])

        if self.num_shared_fcs > 0:
            for h in range(self.class_heads):
                if self.with_avg_pool:
                    x_out[h] = self.avg_pool(x_out[h])

                x_out[h] = x_out[h].flatten(1)

                for fc in self.shared_fcs[h]:
                    x_out[h] = self.relu(fc(x_out[h]))
        
        # separate branches
        x_cls = x_out
        x_reg = x_out
        cls_score=[0 for h in range(self.class_heads)]
        bbox_pred=[0 for h in range(self.class_heads)]
        
        for h in range(self.class_heads):
            for conv in self.cls_convs[h]:
                x_cls[h] = conv(x_cls[h])
            if x_cls[h].dim() > 2:
                if self.with_avg_pool:
                    x_cls[h] = self.avg_pool(x_cls[h])
                x_cls[h] = x_cls[h].flatten(1)
            for fc in self.cls_fcs[h]:
                x_cls[h] = self.relu(fc(x_cls[h]))

            for conv in self.reg_convs[h]:
                x_reg[h] = conv(x_reg[h])
            if x_reg[h].dim() > 2:
                if self.with_avg_pool:
                    x_reg[h] = self.avg_pool(x_reg[h])
                x_reg[h] = x_reg[h].flatten(1)
            for fc in self.reg_fcs[h]:
                x_reg[h] = self.relu(fc(x_reg[h]))

            cls_score[h] = self.fc_cls[h](x_cls[h]) if self.with_cls else None
            bbox_pred[h] = self.fc_reg[h](x_reg[h]) if self.with_reg else None
            
        if self.with_cls:
            cls_score= torch.cat(cls_score,1)
        else:
            cls_score = None
        
        if x_switch.dim() > 2:
            if self.with_avg_pool:
                x_switch = self.avg_pool(x_switch)
            x_switch = x_switch.flatten(1)
            
        switch_logit = self.switch(self.relu(self.hidden_switch[0](x_switch)))
        switch_logit = torch.clamp(switch_logit,min=-12,max=12)
        cls_score = torch.cat([cls_score,switch_logit],1)
        
        reg_prob = switch_logit.sigmoid()
        final_bbox_pred = bbox_pred[0]*reg_prob + bbox_pred[1]*(1.0-reg_prob)
            
        return cls_score, final_bbox_pred


@HEADS.register_module()
class SharedMultiSibling(MultiSibling):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(SharedMultiSibling, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
