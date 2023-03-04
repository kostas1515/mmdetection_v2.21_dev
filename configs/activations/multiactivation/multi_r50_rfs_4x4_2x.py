_base_ = [
    '../../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis_v1.py'
]

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="Icloglog",activation='normal'),
                                         init_cfg = dict(type='Constant',val=0.001, bias=-3.0, override=dict(name='fc_cls')))))

work_dir='./experiments/multi_r50_rfs_4x4_2x/'