_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/segcracks.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (384, 384)
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=False)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        in_channels=3,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=2),
    auxiliary_head=dict(in_channels=384, num_classes=2))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=20000,
        by_epoch=False,
    )
]

# 明确指定使用普通的图像加载器，避免使用需要gdal的遥感图像加载器
train_pipeline = [
    dict(type='LoadImageFromFile'),  # 使用普通图像加载器
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),  # 使用普通图像加载器
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# 更新数据加载器的pipeline并减小batch size以防止OOM
train_dataloader = dict(
    batch_size=2,  # 从4减小到1
    dataset=dict(
        dataset=dict(pipeline=train_pipeline)))

val_dataloader = dict(
    batch_size=1,
    dataset=dict(pipeline=test_pipeline))
    
test_dataloader = val_dataloader
