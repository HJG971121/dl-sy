import os
import time
from omegaconf import OmegaConf

from detectron2.config import LazyCall as L
from detectron2.projects.segmentation.data import (BaseDataset,
                                                   build_train_loader,
                                                   build_test_loader)
from detectron2.projects.segmentation.modeling import MADenseUNetBackbone
# from detectron2.segmentation.modeling.backbone.ma_denseunet_backbone_revision import MADenseUNetBackbone
# from detectron2.segmentation.modeling.backbone.ma_denseunet_backbone_uprevision import MADenseUNetBackbone
from detectron2.projects.segmentation.data import parse_json_annotation_file
from detectron2.projects.segmentation.transforms import FlipTransform, RandomCrop, CenterCrop,RandomSizeCrop, RandomCBTransform, RotateTransform

from ..common.optim import AdamW as optimizer
from ..common.optim import grad_clippers
from ..common.schedule import multi_step_scheduler as lr_scheduler
from ..common.split_combine import SplitCombiner1
# from ..common.split_combine import *

from ..data.data_mapper import GnrtMTTDataMapper
from ..evaluation.evaluator import GnrtMTTEvaluator
from ..modeling.MTT_gnrt import MTTGenerator
from ..modeling.head import MTTGnrtHead
from ..common.losses import (LossList, MSEloss, vMSEloss, L1loss, vL1loss, VGGloss, vDICEloss, vessel_loss)

# ================================================================
# output_dir
# ================================================================
OUTPUT_DIR = 'E:/dl-sy/GenerativeMTT/results'

file_name, _ = os.path.splitext(os.path.basename(__file__))
creat_time = time.strftime('%y%m%d', time.localtime(time.time()))

output_dir = os.path.join(OUTPUT_DIR, f'{file_name}_{creat_time}')
os.makedirs(output_dir, exist_ok=True)

# ================================================================
# 设置 global variable
# ================================================================
# INIT_CHECKPOINT = 'E:\dl-sy\GenerativeMTT\\results\\' \
# 'baseline_MADenseUNet_240218_MSE+vMSE_FullTransformbutCB_D3Picked_ds2_1_finetune\model_0022399.pth'

# 0
# 'baseline_MADenseUNet_240218_MSE+vMSE_FullTransformbutCB_D3Picked_ds2_1_finetune\model_0022399.pth'

# 1
# 'baseline_MADenseUNet_240219_MSE_FullTransformbutCB_D3Picked_ds2_1\model_0047599.pth'

# 2
# 'baseline_MADenseUNet_240229_MSE+vMSE_FullTransformbutCB_D3Picked\model_0016799.pth'




INIT_CHECKPOINT = ''
# INIT_CHECKPOINT = 'E:\dl-sy\GenerativeMTT\\results\\baseline_MADenseUNet_231229_MSE+vMSE_RandomSizeCrop_PickedData_ds2_1\model_0029999.pth'
# INIT_CHECKPOINT = 'E:\dl-sy\GenerativeMTT\\results\\baseline_MADenseUNet_240522_MSE+vMSE_ds2_1_MS_30_finetune\model_0036399.pth'


ANNO_FILE_TRAIN = 'E:\Generative_MTT\data\jsons\GenerativeMTT_picked_LSI_MS_30_train.json'
# ANNO_FILE_VALID = 'E:\Generative_MTT\data\jsons\GenerativeMTT_picked_LSI_MS_30_valid.json'
ANNO_FILE_VALID = 'E:\Generative_MTT\data\jsons\GenerativeMTT_S13_LSI_MS_30_valid.json'
# ANNO_FILE_TRAIN = 'E:\Generative_MTT\data\jsons\GenerativeMTT_picked_LSI_train.json'
# ANNO_FILE_VALID = 'E:\Generative_MTT\data\jsons\GenerativeMTT_picked_LSI_valid.json'
# ANNO_FILE_VALID = 'E:\Generative_MTT\human_brain\data\jsons\GenerativeMTT_human_valid.json'
# ANNO_FILE_VALID = 'E:\Generative_MTT\data\jsons\GenerativeMTT_S15_LSI_valid.json'
# ANNO_FILE_VALID = 'E:\Generative_MTT\data\jsons\GenerativeMTT_picked_LSI_train.json'
# ANNO_FILE_VALID = 'E:\Generative_MTTT\MTTvideo\data\jsons\GenerativeMTT_video_valid.json'
META, DATA_LIST = parse_json_annotation_file(ANNO_FILE_TRAIN)
TRANSFORM_FIELD = {'image': 'image', 'label': 'segmentation'}

# training params
BATCH_SIZE_PER_GPU=10
BATCH_SIZE_PER_GPU_VALID=1
NUM_WORKERS=5
GPU_NUM = 1
DATA_NUM = len(DATA_LIST)

TRAIN_EPOCHS = 100
TRAIN_REPEAT = 200
EPOCH_ITERS = (DATA_NUM * TRAIN_REPEAT) // (GPU_NUM * BATCH_SIZE_PER_GPU)
MAX_ITERS = TRAIN_EPOCHS * EPOCH_ITERS
AMP_ENABLED = True
SAVE_EPOCH = 5
EVAL_EPOCH = 5
LOG_ITER = 5
GRAD_CLIPPER = grad_clippers.grad_value_clipper

# dataloader parameters
# RandomSizeCrop
CROP_SIZE = [256, 256]
RESIZE_SIZE = 256
# RandomFlip
FLIP_AXIS=[1, 1]
FLIP_FREQ=60
# RandomRotate
ROT_ANGLE=[-10,10]
ROT_FREQ=30
# Random CB
CONTRAST_RANGE=[0.8,1.2]
BRIGHTNESS_RANGE=[-50,50]
CB_FREQ=50

# split combine
EVAL_CROP_SIZE = [256, 256]
STRIDE = [128, 128]
SPLIT_COMBINE_ENABLE = True
COMBINE_METHOD = 'gw'
SIGMA = 0.25
SAVE_FEATURES = False
SAVE_CAMS = True
# MARGIN = [8, 8]
# PAD_MODE = 'constant'
# PAD_VALUE = 0


# optimizer parameters
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
LR_VALUES = [0.1, 0.01, 0.001]
LR_MILESTONES = [EPOCH_ITERS*26, EPOCH_ITERS*51, EPOCH_ITERS*76]
WARMUP_ITER = 20

# model parameters
INPUT_CHANNEL = 4
OUTPUT_CHANNEL = 1
FEATS = [x//1 for x in [16, 32, 64, 128, 256]]
NUM_LAYERS = [x for x in [2, 2, 2, 2, 2]]
GROWTH_RATE = [x//1 for x in [8, 16, 32, 64, 128]]
BLOCKS = [2, 4, 5, 6, 6]
DEEP_SUPERVISION = 2
INTERLOSS_WEIGHT = [0.125, 0.125, 0.125, 0.5]
IS_DROP = False

# preprocess
# NORM_PRAMS = [[0.6774833798408508, 0.4883715808391571, 0.4923457205295563, 0.5026843547821045],
#               [0.14446789026260376, 0.13899296522140503, 0.1572393774986267, 0.2881369888782501]]
# # histmatch
# NORM_PRAMS = [[0.6151946187019348, 0.3887929320335388, 0.38793569803237915, 0.5021044611930847],
#               [0.08579732477664948, 0.09849081933498383, 0.11477143317461014, 0.28885218501091003]]
# NORM_PRAMS = [[0.50310165, 0.69311076, 0.51981676, 0.52014786],
#                 [0.2879139, 0.138308, 0.1481872, 0.15913244]]
NORM_PRAMS = [[0] * INPUT_CHANNEL, [1] * INPUT_CHANNEL]
LOSS_FUNCTION = L(LossList)(
    losses=[MSEloss(), vMSEloss(0,'lt')],
    weights=[1, 1]
)

# ================================================================
# 设置 dataloader
# ================================================================
dataloader = OmegaConf.create()

dataloader.train = L(build_train_loader)(
    dataset=L(BaseDataset)(anno_file=ANNO_FILE_TRAIN),
    mapper=L(GnrtMTTDataMapper)(
        transforms=[
            # L(RandomCBTransform)(
            #     contrast_range=CONTRAST_RANGE,
            #     brightness_range=BRIGHTNESS_RANGE,
            #     transform_freq=CB_FREQ,
            #     fields=TRANSFORM_FIELD
            # ),
            L(RotateTransform)(
                rot_angle=ROT_ANGLE,
                rot_freq=ROT_FREQ,
                fields=TRANSFORM_FIELD
            ),
            L(RandomSizeCrop)(
                crop_size = CROP_SIZE,
                fields = TRANSFORM_FIELD
            ),
            L(FlipTransform)(
                flip_axis = FLIP_AXIS,
                flip_freq = FLIP_FREQ,
                fields = TRANSFORM_FIELD
            )

        ],
    ),
    batch_size=BATCH_SIZE_PER_GPU,
    num_workers=NUM_WORKERS,
)

dataloader.test = L(build_test_loader)(
    dataset=L(BaseDataset)(anno_file=ANNO_FILE_VALID),
    mapper=L(GnrtMTTDataMapper)(
        transforms=[
            # L(CenterCrop)(
            #     crop_size = CROP_SIZE,
            #     fields = TRANSFORM_FIELD
            # )
        ],
    ),
    batch_size=BATCH_SIZE_PER_GPU_VALID,
    num_workers=NUM_WORKERS,
)

dataloader.evaluator = [
    L(GnrtMTTEvaluator)(
        output_dir = output_dir,
        checkpoint = INIT_CHECKPOINT,
        save_features = SAVE_FEATURES,
        save_cams = SAVE_CAMS
    )
]

# ================================================================
# 设置 model
# ================================================================
model = L(MTTGenerator)(
        backbone = L(MADenseUNetBackbone)(
            input_channel=INPUT_CHANNEL,
            feats = FEATS,
            scales = [2, 2, 2, 2, 2],
            num_layers = NUM_LAYERS,
            growth_rate = GROWTH_RATE,
            slim=True,
            abn=2,
        ),
        head = L(MTTGnrtHead)(
            loss_function = LOSS_FUNCTION,
            in_channels = FEATS[:-1],
            out_channel = OUTPUT_CHANNEL,
            is_drop = IS_DROP,
            deep_supervision=DEEP_SUPERVISION,
            interloss_weight = INTERLOSS_WEIGHT
        ),
        pixel_mean = NORM_PRAMS[0],
        pixel_std = NORM_PRAMS[1],
        resize_size = RESIZE_SIZE,
        output_dir=output_dir,
        save_features = SAVE_FEATURES,
        save_cams = SAVE_CAMS
)


# ================================================================
# 设置 optimizer 和 scheduler
# ================================================================
optimizer.lr = LEARNING_RATE
optimizer.weight_decay = WEIGHT_DECAY
optimizer.eps = 1e-6

# multi step scheduler
lr_scheduler.values = LR_VALUES
lr_scheduler.milestones = LR_MILESTONES

# cosine step scheduler
# lr_scheduler.start = LEARNING_RATE
# lr_scheduler.end = LEARNING_RATE * 0.001

lr_scheduler.max_iter = MAX_ITERS
lr_scheduler.warmup_iter = WARMUP_ITER


# ================================================================
# 设置 train
# ================================================================
train=dict(
    output_dir=output_dir,
    init_checkpoint=INIT_CHECKPOINT,
    max_iter=MAX_ITERS,
    amp=dict(
        enabled=AMP_ENABLED,
        grad_clipper=GRAD_CLIPPER,
    ),
    ddp=dict(
        broadcast_buffer=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    checkpointer=dict(
        period=EPOCH_ITERS * SAVE_EPOCH,
        max_to_keep=100,
    ),
    split_combine=dict(
        enabled=SPLIT_COMBINE_ENABLE,
        # split_combiner=L(BaseSplitCombiner)(
        #     margin = MARGIN,
        #     crop_size = CROP_SIZE,
        #     pad_value = PAD_VALUE,
        #     pad_mode = PAD_MODE,
        #     combine_method=COMBINE_METHOD,
        #     device='cpu'
        # )
        # split_combiner=L(SplitCombiner)(
        #     crop_size=CROP_SIZE,
        #     combine_method=COMBINE_METHOD,
        #     device='cpu'
        # )
        split_combiner=L(SplitCombiner1)(
            crop_size=EVAL_CROP_SIZE,
            stride = STRIDE,
            combine_method=COMBINE_METHOD,
            save_features = SAVE_FEATURES,
            save_cams = SAVE_CAMS,
            device='cuda',
            sigma = SIGMA
        )
    ),
    eval_period=EPOCH_ITERS * EVAL_EPOCH,
    log_period=LOG_ITER,
    device='cuda',
)
