_base_ = [
    '../../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
]

data = dict(train=dict(oversample_thr=0.0))


model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="GumbelFocalLoss",num_classes=1203,gamma=2.0,alpha=0.25,use_sigmoid=True),
                                         init_cfg = dict(type='Constant',val=0.001, bias=-2.0, override=dict(name='fc_cls')))))

work_dir='./experiments/gumbel_fl_r50_4x4_1x/'
# work_dir='./experiments/test/'

# get_stats=1