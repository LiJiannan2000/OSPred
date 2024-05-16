import argparse
import sys
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from core.config import config
from core.scheduler import PolyScheduler
from core.loss import CoxLoss
from core.function import inference_Brats20
from utils.utils import determine_device, save_checkpoint, update_config, create_logger, setup_seed
from models.model import OSnet
from dataset.dataloader import get_dataset_BraTS20


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='',
                        help='path for pretrained weights', type=str)
    parser.add_argument('--fold', default=0, help='which data fold to train on', type=int)
    parser.add_argument('--dir', default='', help='path suffix', type=str)
    args = parser.parse_args()
    return args


def main(args):
    net = OSnet
    devices = config.TRAIN.DEVICES
    model = net("learned")
    model = nn.DataParallel(model, devices).cuda()

    validset = get_dataset_BraTS20('val')

    best_perf = 0.0
    logger = create_logger('log' + args.dir, 'test' + str(args.fold) + '.log')

    c_index = []
    for epoch in range(100):
        ld_model = 'experiments' + args.dir + '/5fold' + str(args.fold) + '_checkpoint_' + str(epoch) + '.pth'
        checkpoint = torch.load(ld_model)
        model.load_state_dict(checkpoint['state_dict'])

        logger.info(f'Epoch: [{epoch}]')
        perf = inference_Brats20(model, validset, logger, config, best_perf)
        c_index.append(perf)
        if perf > best_perf:
            best_perf = perf


if __name__ == '__main__':
    args = parse_args()
    main(args)
