_base_ = [
    '../_base_/models/fcn_r50-d8.py',
    '../_base_/datasets/segcracks.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (769, 769)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(dilations=(1, 1, 1, 2), strides=(1, 2, 2, 1)),
    decode_head=dict(align_corners=True, dilation=6, num_classes=2),
    auxiliary_head=dict(align_corners=True, dilation=6, num_classes=2),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))

# Update the train pipeline to ensure consistent sizes
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # Force exact size without maintaining aspect ratio
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# Update the test pipeline to ensure consistent sizes
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # Force exact size without maintaining aspect ratio
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# Update dataloaders with new pipeline configurations - corrected for nested structure
train_dataloader = dict(
    dataset=dict(
        dataset=dict(pipeline=train_pipeline)
    )
)
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# Update data preprocessor with valid parameters
data_preprocessor.update(dict(
    size=crop_size,
    pad_val=0,
    seg_pad_val=255
))
