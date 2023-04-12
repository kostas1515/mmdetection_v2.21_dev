from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer
from mmcv.ops.carafe import CARAFEPack
from mmcv.runner import ModuleList, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import mask_target
from mmdet.models.builder import HEADS, build_loss

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 2048**2  # 1 GB memory limit

@HEADS.register_module()
class MultiFCNMaskHead(nn.Module):
    def __init__(self,
                 num_convs=3,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=80,
                 class_agnostic=False,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 conv_cfg=None,
                 norm_cfg=None,
                 predictor_cfg=dict(type='Conv'),
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True,activated=True, loss_weight=1.0),
                 class_heads=2):
        super(MultiFCNMaskHead, self).__init__()
        self.class_heads=class_heads
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
                None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                f'Invalid upsample method {self.upsample_cfg["type"]}, '
                'accepted methods are "deconv", "nearest", "bilinear", '
                '"carafe"')
        self.num_convs = num_convs
        self.predictor_cfg = predictor_cfg
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None)
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask)
        
        
        self.convs =  nn.ModuleList()
        
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            for head in range(self.class_heads):
                self.convs.append(
                    ConvModule(
                        in_channels,
                        self.conv_out_channels,
                        self.conv_kernel_size,
                        padding=padding,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg))
        
#         self.switch = nn.ConvTranspose2d(
#                     in_channels,
#                     1,
#                     kernel_size=self.scale_factor,
#                     stride=self.scale_factor).cuda()

        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        upsample_cfg_ = self.upsample_cfg.copy()
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            upsample_cfg_.update(
                in_channels=upsample_in_channels,
                out_channels=self.conv_out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor)
            self.upsample = nn.ModuleList([build_upsample_layer(upsample_cfg_) for head in range(self.class_heads)])
        elif self.upsample_method == 'carafe':
            upsample_cfg_.update(
                channels=upsample_in_channels, scale_factor=self.scale_factor)
            self.upsample = nn.ModuleList([build_upsample_layer(upsample_cfg_) for head in range(self.class_heads)])
        else:
            # suppress warnings
            align_corners = (None
                             if self.upsample_method == 'nearest' else False)
            upsample_cfg_.update(
                scale_factor=self.scale_factor,
                mode=self.upsample_method,
                align_corners=align_corners)
            self.upsample = nn.ModuleList([build_upsample_layer(upsample_cfg_) for head in range(self.class_heads)])

        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits = nn.ModuleList([build_conv_layer(self.predictor_cfg,
                                            logits_in_channel, out_channels, 1) for head in range(self.class_heads)])
        
        self.switch = nn.Sequential(ConvModule(
                        in_channels,
                        self.conv_out_channels,
                        self.conv_kernel_size,
                        padding=padding,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg),build_upsample_layer(upsample_cfg_),build_conv_layer(dict(type='Conv'),
                                            logits_in_channel, 1, 1)).cuda()
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None
        
    @auto_fp16()
    def forward(self, x):
        switch_logit  = self.switch(x)
        switch_logit = torch.clamp(switch_logit,min=-12,max=12)
        x_out=[x for head in range(self.class_heads)]
        for head,conv in enumerate(self.convs):
            x_out[head % self.class_heads]=conv(x_out[head % self.class_heads])
            # x = conv(x)
        
        if self.upsample is not None:
            for head in range(self.class_heads):
                x_out[head] = self.upsample[head](x_out[head])
                if self.upsample_method == 'deconv':
                    x_out[head] = self.relu(x_out[head])
        
        gumbits = self.conv_logits[0](x_out[0])
        normits = self.conv_logits[1](x_out[1])
        
        pestim_g = 1/(torch.exp(torch.exp(-(torch.clamp(gumbits,min=-4,max=10)))))
        pestim_n=1/2+torch.erf(torch.clamp(normits,min=-5,max=6)/(2**(1/2)))/2
        probs = switch_logit.sigmoid()
        
        mask_pred = probs*pestim_g+(1-probs)*pestim_n
#         mask_pred = torch.clamp(mask_pred,min=0.0001,max=0.9999)
        return mask_pred
    
    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets
    
    def init_weights(self):
        for module in [self.upsample, self.conv_logits]:
            for m in module:
                if m is None:
                    continue
                elif isinstance(m, CARAFEPack):
                    m.init_weights()
                elif hasattr(m, 'weight') and hasattr(m, 'bias'):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.bias,1.0)
                    
    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels):
        """
        Example:
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> # There are lots of variations depending on the configuration
            >>> self = FCNMaskHead(num_classes=C, num_convs=1)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_pred = self.forward(inputs)
            >>> sf = self.scale_factor
            >>> labels = torch.randint(0, C, size=(N,))
            >>> # With the default properties the mask targets should indicate
            >>> # a (potentially soft) single-class label
            >>> mask_targets = torch.rand(N, H * sf, W * sf)
            >>> loss = self.loss(mask_pred, mask_targets, labels)
            >>> print('loss = {!r}'.format(loss))
        """
        loss = dict()
        if mask_pred.size(0) == 0:
            loss_mask = mask_pred.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_pred, mask_targets,
                                           torch.zeros_like(labels))
            else:
                loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss
    

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape (Tuple): original image height and width, shape (2,)
            scale_factor(ndarray | Tensor): If ``rescale is True``, box
                coordinates are divided by this scale factor to fit
                ``ori_shape``.
            rescale (bool): If True, the resulting masks will be rescaled to
                ``ori_shape``.

        Returns:
            list[list]: encoded masks. The c-th item in the outer list
                corresponds to the c-th class. Given the c-th outer list, the
                i-th item in that inner list is the mask for the i-th box with
                class label c.

        Example:
            >>> import mmcv
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> self = FCNMaskHead(num_classes=C, num_convs=0)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_pred = self.forward(inputs)
            >>> # Each input is associated with some bounding box
            >>> det_bboxes = torch.Tensor([[1, 1, 42, 42 ]] * N)
            >>> det_labels = torch.randint(0, C, size=(N,))
            >>> rcnn_test_cfg = mmcv.Config({'mask_thr_binary': 0, })
            >>> ori_shape = (H * 4, W * 4)
            >>> scale_factor = torch.FloatTensor((1, 1))
            >>> rescale = False
            >>> # Encoded masks are a list for each category.
            >>> encoded_masks = self.get_seg_masks(
            >>>     mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape,
            >>>     scale_factor, rescale
            >>> )
            >>> assert len(encoded_masks) == C
            >>> assert sum(list(map(len, encoded_masks))) == N
        """

        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)
                     ]  # BG is not included in num_classes
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        # In most cases, scale_factor should have been
        # converted to Tensor when rescale the bbox
        if not isinstance(scale_factor, torch.Tensor):
            if isinstance(scale_factor, float):
                scale_factor = np.array([scale_factor] * 4)
                warn('Scale_factor should be a Tensor or ndarray '
                     'with shape (4,), float would be deprecated. ')
            assert isinstance(scale_factor, np.ndarray)
            scale_factor = torch.Tensor(scale_factor)

        if rescale:
            img_h, img_w = ori_shape[:2]
            bboxes = bboxes / scale_factor.to(bboxes)
        else:
            w_scale, h_scale = scale_factor[0], scale_factor[1]
            img_h = np.round(ori_shape[0] * h_scale.item()).astype(np.int32)
            img_w = np.round(ori_shape[1] * w_scale.item()).astype(np.int32)

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            # the types of img_w and img_h are np.int32,
            # when the image resolution is large,
            # the calculation of num_chunks will overflow.
            # so we need to change the types of img_w and img_h to int.
            # See https://github.com/open-mmlab/mmdetection/pull/5191
            num_chunks = int(
                np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT /
                        GPU_MEM_LIMIT))
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8)

        if not self.class_agnostic:
            mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk

        for i in range(N):
            cls_segms[labels[i]].append(im_mask[i].detach().cpu().numpy())
        return cls_segms

    def onnx_export(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, **kwargs):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor): shape (n, #class, h, w).
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape (Tuple): original image height and width, shape (2,)

        Returns:
            Tensor: a mask of shape (N, img_h, img_w).
        """

        bboxes = det_bboxes[:, :4]
        labels = det_labels
        # No need to consider rescale and scale_factor while exporting to ONNX
        img_h, img_w = ori_shape[:2]
        threshold = rcnn_test_cfg.mask_thr_binary
        if not self.class_agnostic:
            box_inds = torch.arange(mask_pred.shape[0])
            mask_pred = mask_pred[box_inds, labels][:, None]
        masks, _ = _do_paste_mask(
            mask_pred, bboxes, img_h, img_w, skip_empty=False)
        if threshold >= 0:
            # should convert to float to avoid problems in TRT
            masks = (masks >= threshold).to(dtype=torch.float)
        return masks
    
def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks according to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(
            boxes.min(dim=0).values.floor()[:2] - 1,
            min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(
            boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(
            boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device).to(torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device).to(torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    # IsInf op is not supported with ONNX<=1.7.0
    if not torch.onnx.is_in_onnx_export():
        if torch.isinf(img_x).any():
            inds = torch.where(torch.isinf(img_x))
            img_x[inds] = 0
        if torch.isinf(img_y).any():
            inds = torch.where(torch.isinf(img_y))
            img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()