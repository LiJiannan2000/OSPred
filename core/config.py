from yacs.config import CfgNode as CN

config = CN()
config.NUM_WORKERS = 6
config.PRINT_FREQ = 5
config.VALIDATION_INTERVAL = 5
config.OUTPUT_DIR = 'experiments'
config.SEED = 12345

config.CUDNN = CN()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

config.DATASET = CN()
config.DATASET.ROOT = 'DATA/zhengdayi_123_all'
config.DATASET.DEF_ROOT = 'DATA/zhengdayi_123_radiomics_deformation'
config.DATASET.TEST_ROOT = 'DATA/BraTS20_all'
config.DATASET.TEST_DEF_ROOT = 'DATA/BraTS20_radiomics_deformation'
config.DATASET.input_channel = 16
config.DATASET.QUEUE_LENGTH = 300
config.DATASET.SAMPLES_PER_VOLUME = 10


config.MODEL = CN()
config.MODEL.NAME = 'OSPred'
config.MODEL.USE_PRETRAINED = True
config.MODEL.PRETRAINED = ''
config.MODEL.EXTRA = CN(new_allowed=True)
config.MODEL.INPUT_SIZE = [140, 168, 140]
config.MODEL.FEATURE_DEF_SIZE = 240
config.MODEL.NUM_HEADS = 4
config.MODEL.NUM_LAYERS = 2
config.MODEL.DROPOUT_RATE = 0.1
config.MODEL.ATTN_DROPOUT_RATE = 0.1

config.TRAIN = CN()
config.TRAIN.logdir = 'runs'
config.TRAIN.LR = 1e-4
config.TRAIN.WEIGHT_DECAY = 1e-4
config.TRAIN.WEIGHT_DISTILL_DEF = 0.5
config.TRAIN.WEIGHT_DISTILL_FEA = 0.5
config.TRAIN.BATCH_SIZE = 16
config.TRAIN.NUM_BATCHES = 250
config.TRAIN.EPOCH = 100
config.TRAIN.DEVICES = [0, 1, 2, 3]

config.INFERENCE = CN()
config.INFERENCE.BATCH_SIZE = 16

