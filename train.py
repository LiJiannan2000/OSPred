import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from core.config import config
from core.scheduler import PolyScheduler
from core.loss import CoxLoss, distillation_loss
from core.function import train, inference
from utils.utils import determine_device, save_checkpoint, update_config, create_logger, setup_seed
from models.model import OSnet
from dataset.dataloader import get_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='', help='path for pretrained weights', type=str)
    parser.add_argument('--fold', default=0, help='which data fold to train on', type=int)
    parser.add_argument('--logdir', default='log', help='path for log', type=str)
    parser.add_argument('--runsdir', default='runs', help='path for runs', type=str)
    parser.add_argument('--outdir', default='experiments', help='path for experiments', type=str)
    parser.add_argument('--dir', default='', help='path suffix', type=str)
    args = parser.parse_args()
    return args


def main(args):
    setup_seed(config.SEED)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    net = OSnet  # use your network architecture here --> <file_name>.<class_name>
    devices = config.TRAIN.DEVICES
    teacher = net("learned", True)
    teacher = nn.DataParallel(teacher, devices).cuda()
    # load teacher pretrained weights
    teacherweights = torch.load('teacher_weights/' + str(args.fold))
    teacher.load_state_dict(teacherweights['state_dict'])

    model = net("learned")
    model = nn.DataParallel(model, devices).cuda()
    optimizer = optim.SGD(model.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY, momentum=0.9, nesterov=True)
    scheduler = PolyScheduler(optimizer, t_total=config.TRAIN.EPOCH)
    criterion_MSE = nn.MSELoss()
    criterion = CoxLoss()

    train_dataset = get_dataset(args.fold, 'train', config.DATASET.input_channel)
    valid_dataset = get_dataset(args.fold, 'val', config.DATASET.input_channel)

    best_perf = 0.0
    logger = create_logger(args.logdir + args.dir, 'train' + str(args.fold) + '.log')
    with open('core/config.py', 'r') as f:
        config_contents = f.read()
    logger.info('Config contents: \n%s', config_contents)

    writer = SummaryWriter(log_dir=args.runsdir + args.dir + str(args.fold))

    for epoch in range(config.TRAIN.EPOCH):
        logger.info('learning rate : {}'.format(optimizer.param_groups[0]['lr']))

        train(teacher, model, train_dataset, optimizer, criterion, distillation_loss, logger, config, epoch, writer)
        scheduler.step()
        # running validation at every epoch is time consuming
        perf = inference(model, valid_dataset, criterion, logger, config, best_perf, epoch, writer)

        if perf > best_perf:
            best_perf = perf
            best_model = True
        else:
            best_model = False

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'c-index': perf,
            'optimizer': optimizer.state_dict(),
        }, best_model, args.outdir + args.dir, args.fold, filename=str(args.fold) + '_checkpoint_' + str(epoch) + '.pth')
    writer.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
