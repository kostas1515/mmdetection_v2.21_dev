_base_ = './retinanet_r50_fpn_1x_coco_gumbel.py'
# learning policy
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)

work_dir='./experiments/coco/retinanet_r50_fpn_2x_coco_gumbel/'
