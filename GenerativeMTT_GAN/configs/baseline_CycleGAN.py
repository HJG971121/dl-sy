import os
import time
from omegaconf import OmegaConf

from detectron2.config import LazyCall as L
from detectron2.projects.segmentation.data import (BaseDataset,
                                                   build_train_loader,
                                                   build_test_loader)
from detectron2.projects.generation.modeling.generator import UNetBackbone
from detectron2.projects.generation.modeling.discriminator import NLayerDiscriminator
from detectron2.projects.segmentation.modeling import MADenseUNetBackbone
# from detectron2.segmentation.modeling.backbone.ma_denseunet_backbone_revision import MADenseUNetBackbone
# from detectron2.segmentation.modeling.backbone.ma_denseunet_backbone_uprevision import MADenseUNetBackbone
from detectron2.projects.segmentation.data import parse_json_annotation_file
from detectron2.projects.segmentation.transforms import FlipTransform, RandomSizeCrop
from detectron2.projects.segmentation.transforms.split_combine import SplitCombiner1

from ..common.optim import AdamW as optimizer
from ..common.optim import grad_clippers
from ..common.schedule import multi_step_scheduler as lr_scheduler
# from ..common.split_combine import *

from ..data.data_mapper import GnrtMTTGANDataMapper
from ..evaluation.evaluator import GnrtMTTEvaluator
from ..modeling.MTTgnrtCycleGAN import MTTGeneratorCycleGAN
from ..common.losses import (LossList, MSEloss, vMSEloss, L1loss, GANLoss)

# ================================================================
# output_dir
# ================================================================
OUTPUT_DIR = 'E:/dl-sy/GenerativeMTT_GAN/results'

file_name, _ = os.path.splitext(os.path.basename(__file__))
creat_time = time.strftime('%y%m%d', time.localtime(time.time()))

output_dir = os.path.join(OUTPUT_DIR, f'{file_name}_{creat_time}')
os.makedirs(output_dir, exist_ok=True)

# ================================================================
# 设置 global variable
# ================================================================
INIT_CHECKPOINT = ''

ANNO_FILE_TRAIN = 'E:\Generative_MTT\data\jsons\GenerativeMTT_picked_train.json'
ANNO_FILE_VALID = 'E:\Generative_MTT\data\jsons\GenerativeMTT_picked_train.json'
META, DATA_LIST = parse_json_annotation_file(ANNO_FILE_TRAIN)
TRANSFORM_FIELD = {'image_A': 'image', 'image_B': 'segmentation', 'v_mask': 'segmentation'}
# training params
BATCH_SIZE_PER_GPU=10
BATCH_SIZE_PER_GPU_VALID=1
NUM_WORKERS=10
GPU_NUM = 1
DATA_NUM = len(DATA_LIST)

TRAIN_EPOCHS = 100
TRAIN_REPEAT = 200
EPOCH_ITERS = (DATA_NUM * TRAIN_REPEAT) // (GPU_NUM * BATCH_SIZE_PER_GPU)
MAX_ITERS = TRAIN_EPOCHS * EPOCH_ITERS
AMP_ENABLED = True
SAVE_EPOCH = 10
EVAL_EPOCH = 101
LOG_ITER = 5
GRAD_CLIPPER = grad_clippers.grad_value_clipper

# dataloader parameters
CROP_SIZE = [256, 256]
EVAL_CROP_SIZE = [256, 256]
RESIZE_SIZE = 256
MARGIN = [8, 8]
STRIDE = [32, 32]
SPLIT_COMBINE_ENABLE = True
COMBINE_METHOD = 'gw'
SIGMA = 0.5
FLIP_AXIS=[1, 1]
FLIP_FREQ=60
ROT_ANGLE=[-10,10]
ROT_FREQ=50

# optimizer parameters
OPTIM_PARA = {
    'lr': 0.0002,
    'weight_decay': 0.001,
    'beta1': 0.9
}

LR_VALUES = [0.1, 0.01, 0.001]
LR_MILESTONES = [EPOCH_ITERS*26, EPOCH_ITERS*51, EPOCH_ITERS*76]
WARMUP_ITER = 20

# model parameters
INPUT_CHANNEL = 1
OUTPUT_CHANNEL = 1
FEATS = [x//1 for x in [16, 32, 64, 128, 256]]
FEATS_D = [32, 64, 128]
NUM_LAYERS = [2, 2, 2, 2, 2]
GROWTH_RATE = [x//1 for x in [8, 16, 32, 64, 128]]
BLOCKS = [2, 4, 5, 6, 6]
DEEP_SUPERVISION = 2
INTERLOSS_WEIGHT = [0.125, 0.125, 0.125, 0.5]
IS_DROPOUT = False
ABN = 2
POOL_SIZE = 50

NORM_PRAMS = [[0]*INPUT_CHANNEL, [1]*INPUT_CHANNEL]
LOSS_FUNCTION = {
    'criterionGAN': GANLoss(),
    'criterionCycle': L1loss(),
    'criterionIdt': L1loss()
}
LOSS_WEIGHT = {
    'labmda_idt': 0,
    'lambda_A': 1,
    'lambda_B': 1
}

# ================================================================
# 设置 dataloader
# ================================================================
dataloader = OmegaConf.create()

dataloader.train = L(build_train_loader)(
    dataset=L(BaseDataset)(anno_file=ANNO_FILE_TRAIN),
    mapper=L(GnrtMTTGANDataMapper)(
        transforms=[
            # L(RotateTransform)(
            #     rot_angle=ROT_ANGLE,
            #     rot_freq=ROT_FREQ,
            #     fields=TRANSFORM_FIELD
            # ),
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
    mapper=L(GnrtMTTGANDataMapper)(
        transforms=[
            # L(RandomCrop)(
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
    )
]

# ================================================================
# 设置 model
# ================================================================
model = L(MTTGeneratorCycleGAN)(
        netG_A = L(UNetBackbone)(
            input_channel=INPUT_CHANNEL,
            output_channel = OUTPUT_CHANNEL,
            feats = FEATS,
            blocks = BLOCKS,
            slim=True,
            is_dropout = IS_DROPOUT,
            abn=ABN,
        ),
        netG_B=L(UNetBackbone)(
            input_channel=OUTPUT_CHANNEL,
            output_channel=INPUT_CHANNEL,
            feats=FEATS,
            blocks=BLOCKS,
            slim=True,
            is_dropout=IS_DROPOUT,
            abn=ABN,
        ),
        netD_A = L(NLayerDiscriminator)(
            in_channel = INPUT_CHANNEL,
            feats = FEATS_D,
            abn=ABN
        ),
        netD_B=L(NLayerDiscriminator)(
            in_channel=OUTPUT_CHANNEL,
            feats = FEATS_D,
            abn=ABN
        ),
        pool_size = POOL_SIZE,
        loss_function = LOSS_FUNCTION,
        weight = LOSS_WEIGHT,
        optim_para = OPTIM_PARA,
        pixel_mean = NORM_PRAMS[0],
        pixel_std = NORM_PRAMS[1],
        resize_size = RESIZE_SIZE,
        output_dir=output_dir,
)

# ================================================================
# 设置 optimizer 和 scheduler
# ================================================================
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
        split_combiner=L(SplitCombiner1)(
            crop_size=EVAL_CROP_SIZE,
            stride = STRIDE,
            combine_method=COMBINE_METHOD,
            device='cpu',
            sigma = SIGMA
        )
    ),
    eval_period=EPOCH_ITERS * EVAL_EPOCH,
    log_period=LOG_ITER,
    device='cuda',
)