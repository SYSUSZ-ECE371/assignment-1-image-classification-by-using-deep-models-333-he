
custom_imports = dict(
    imports=['mmpretrain.models.classifiers'],  # 替换为你的实际模块路径
    allow_failed_imports=False
)
_base_ = [
    '/Users/niezhiqi/Desktop/课程/深度学习/mmpretrain/configs/_base_/models/resnet18.py',  # 继承ImageNet预训练的ResNet50配置
    '/Users/niezhiqi/Desktop/课程/深度学习/mmpretrain/configs/_base_/default_runtime.py'  # 默认运行配置
    #'/Users/niezhiqi/Desktop/课程/深度学习/mmpretrain/configs/_base_/schedules/imagenet_bs256.py'
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



train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)
val_cfg = dict()
test_cfg = None
# 3. 修改学习率策略
# 5. 优化器与学习率
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam',
        lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=0.0001
    )
)
# 7. 学习率调度
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=5,  # 5个epoch的warmup
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=25,  # 总epochs(30) - warmup_epochs(5)
        by_epoch=True,
        begin=5,
        end=30,
    )
]


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


# 4. 配置预训练模型
# 7. 评估器配置（关键修改点）
val_evaluator = dict(
    metrics=[
        # 验证准确率（原有配置）
        dict(type='Accuracy', topk=(1, 5)),
        # 新增验证损失计算
        dict(type='Loss'),  # 关键点：指定name为loss_val以区分训练损失
    ]
)

train_evaluator = dict(type='Accuracy', topk=(1, ))
# 8. 日志和检查点配置（关键修改点）
default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=10,  # 每10次迭代记录一次日志
        log_metric_by_epoch=True  # 按epoch记录指标
    ),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='accuracy/top1',  # 根据验证集Top1保存最佳模型
        rule='greater'
    )
)
# 9. 可视化配置（可选）
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)


load_from = 'checkpoints/resnet18_8xb32_in1k_20210831-fbbb1da6.pth'  # 从Model Zoo下载的权重

