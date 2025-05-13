
custom_imports = dict(
    imports=['mmpretrain.models.classifiers'],  # 替换为你的实际模块路径
    allow_failed_imports=False
)
_base_ = [
    '/Users/niezhiqi/Desktop/课程/深度学习/mmpretrain/configs/_base_/models/shufflenet_v1_1x.py',  # 继承ImageNet预训练的ResNet50配置
    '/Users/niezhiqi/Desktop/课程/深度学习/mmpretrain/configs/_base_/default_runtime.py' , # 默认运行配置
    '/Users/niezhiqi/Desktop/课程/深度学习/mmpretrain/configs/_base_/schedules/imagenet_bs256.py'
]

# 1. 修改模型头适配花朵分类
model = dict(
    head=dict(

        num_classes=5,  # 花朵数据集的5个类别
        topk=(1,),  # 只评估top-1准确率
    ))

# 2. 修改数据集配置
data = dict(
    samples_per_gpu=32,  # 批量大小
    workers_per_gpu=2,   # 数据加载线程数
    train=dict(
        type='CustomDataset',
        data_prefix='/Users/niezhiqi/Desktop/课程/深度学习/hw1/EX1/processed_flower_dataset/train',  # 训练集路径
        ann_file='/Users/niezhiqi/Desktop/课程/深度学习/hw1/EX1/train.txt',  # 训练标注文件
    ),
    val=dict(
        type='CustomDataset',
        data_prefix='/Users/niezhiqi/Desktop/课程/深度学习/hw1/EX1/processed_flower_dataset/val',  # 验证集路径
        ann_file='/Users/niezhiqi/Desktop/课程/深度学习/hw1/EX1/val.txt',  # 验证标注文件
    ))




# 3. 修改学习率策略
# 5. 优化器与学习率
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)

train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)
val_cfg = dict()
# 5. 运行器配置
runner = dict(
    type='EpochBasedRunner',
    max_epochs=30,
    # 显式指定 train_cfg 和 val_cfg
    train_cfg = dict(),
    val_cfg=dict(),
)

train_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='/Users/niezhiqi/Desktop/课程/深度学习/hw1/EX1/processed_flower_dataset/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=224, type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))


val_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='/Users/niezhiqi/Desktop/课程/深度学习/hw1/EX1/processed_flower_dataset/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

test_dataloader = None
test_cfg = None
test_evaluator = None

val_evaluator = dict(type='Accuracy', topk=(1, ))
# 4. 配置预训练模型
load_from = 'checkpoints/shufflenet_v2_batch1024_imagenet_20200812-5bf4721e.pth'  # 从Model Zoo下载的权重