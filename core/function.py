import os
import torch
import numpy as np
import torch.nn as nn
import torchio as tio
from torch.utils.data.dataloader import DataLoader
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes

from utils.utils import AverageMeter, determine_device, get_bbox
from core.loss import CoxLoss


def concordance_index(risk, ostime, death):
    # ground truth matrix
    gt_mat = torch.triu(torch.ones(len(ostime), len(ostime))).cuda(device=risk.device)
    gt_mat = gt_mat - torch.diag_embed(torch.diag(gt_mat))
    # pred matrix
    pred_mat = torch.zeros_like(gt_mat)
    for i in range(len(risk)):
        for j in range(len(risk)):
            if death[i] == 0:
                gt_mat[i, j] = 0
                pred_mat[i, j] = 0
            else:
                if risk[i] > risk[j]:
                    pred_mat[i, j] = 1
                elif risk[i] == risk[j]:
                    pred_mat[i, j] = 1 if ostime[i] == ostime[j] else 0.5
                else:
                    pred_mat[i, j] = 0
    # c_index
    if torch.sum(gt_mat) == 0:
        return torch.sum(gt_mat)
    c_index = torch.sum(pred_mat * gt_mat) / torch.sum(gt_mat)

    return c_index


def train(teacher, model, train_dataset, optimizer, criterion, criterion_distill, logger, config, epoch, writer):
    model.train()
    cox_losses = AverageMeter()
    distill_losses1 = AverageMeter()
    distill_losses2 = AverageMeter()
    distill_losses3 = AverageMeter()
    distill_losses4 = AverageMeter()
    distill_losses_def = AverageMeter()
    distill_losses = AverageMeter()
    losses = AverageMeter()
    scaler = torch.cuda.amp.GradScaler()
    loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )
    print("train data size:", len(train_dataset))
    risks, ostimes, deaths = [], [], []
    for idx, data_dict in enumerate(loader):
        data = data_dict['data']
        data_def = data_dict['data_def']
        death = data_dict['death']
        ostime = data_dict['ostime']
        info = data_dict['info']

        data = data.cuda()
        data_def = data_def.cuda()
        death = death.cuda()
        ostime = ostime.cuda()
        info = info.cuda()
        # run training
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                teacher_out1, teacher_out2, teacher_out3, teacher_out4 = teacher(data, data_def, info)
                teacher_out1 = teacher_out1.squeeze()
                teacher_out2 = teacher_out2.squeeze()
                teacher_out3 = teacher_out3.squeeze()
                teacher_out4 = teacher_out4.squeeze()
                data_ti = torch.chunk(data, 4, dim=1)
                data1 = data_ti[0]
            def_post, out1, out2, out3, risk = model(data1, data_def, info)
            def_post = def_post.squeeze()
            out1 = out1.squeeze()
            out2 = out2.squeeze()
            out3 = out3.squeeze()
            risk = risk.squeeze()
            ostime, indices = torch.sort(ostime)
            death = death[indices]
            teacher_out1 = teacher_out1[indices]
            teacher_out2 = teacher_out2[indices]
            teacher_out2 = teacher_out2[indices]
            teacher_out4 = teacher_out4[indices]
            def_post = info[indices]
            out1 = out1[indices]
            out2 = out2[indices]
            out3 = out3[indices]
            risk = risk[indices]
            loss1 = criterion(risk, ostime, death)
            loss_distill = criterion_distill(def_post, out1, out2, out3, risk, data_def, teacher_out1, teacher_out2,
                                             teacher_out3, teacher_out4)
            loss_distill_def = loss_distill[0]
            loss_distill1 = loss_distill[1]
            loss_distill2 = loss_distill[2]
            loss_distill3 = loss_distill[3]
            loss_distill4 = loss_distill[4]
            loss_distill_f = sum(loss_distill[1:])
            loss_distill = config.TRAIN.WEIGHT_DISTILL_DEF * loss_distill_def + config.TRAIN.WEIGHT_DISTILL_FEA * loss_distill_f
            loss = loss1 + loss_distill
        risks.append(risk)
        ostimes.append(ostime)
        deaths.append(death)
        cox_losses.update(loss1.item(), config.TRAIN.BATCH_SIZE)
        distill_losses_def.update(loss_distill_def.item(), config.TRAIN.BATCH_SIZE)
        distill_losses1.update(loss_distill1.item(), config.TRAIN.BATCH_SIZE)
        distill_losses2.update(loss_distill2.item(), config.TRAIN.BATCH_SIZE)
        distill_losses3.update(loss_distill3.item(), config.TRAIN.BATCH_SIZE)
        distill_losses4.update(loss_distill4.item(), config.TRAIN.BATCH_SIZE)
        distill_losses.update(loss_distill.item(), config.TRAIN.BATCH_SIZE)
        losses.update(loss.item(), config.TRAIN.BATCH_SIZE)
        # do back-propagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        scaler.step(optimizer)
        scaler.update()

        if idx % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Cox Loss: {cox_loss.val:.3f} ({cox_loss.avg:.3f})\t' \
                  'Distill LossDef: {distill_losses_def.val:.3f} ({distill_losses_def.avg:.3f})\t' \
                  'Distill Loss1: {distill_loss1.val:.3f} ({distill_loss1.avg:.3f})\t' \
                  'Distill Loss2: {distill_loss2.val:.3f} ({distill_loss2.avg:.3f})\t' \
                  'Distill Loss3: {distill_loss3.val:.3f} ({distill_loss3.avg:.3f})\t' \
                  'Distill Loss4: {distill_loss4.val:.3f} ({distill_loss4.avg:.3f})\t' \
                  'Distill Loss: {distill_loss.val:.3f} ({distill_loss.avg:.3f})\t' \
                  'Loss: {loss.val:.3f} ({loss.avg:.3f})'.format(epoch, idx, len(loader), cox_loss=cox_losses,
                                                                 distill_losses_def=distill_losses_def,
                                                                 distill_loss1=distill_losses1,
                                                                 distill_loss2=distill_losses2,
                                                                 distill_loss3=distill_losses3,
                                                                 distill_loss4=distill_losses4,
                                                                 distill_loss=distill_losses, loss=losses)
            logger.info(msg)
    if risks[-1].shape:
        risks = torch.cat(risks)
        ostimes = torch.cat(ostimes)
        deaths = torch.cat(deaths)
    else:
        risks = torch.cat(risks[:-1])
        ostimes = torch.cat(ostimes[:-1])
        deaths = torch.cat(deaths[:-1])
    ostimes, indices = torch.sort(ostimes)
    risks = risks[indices]
    deaths = deaths[indices]
    c_index = concordance_index(risks, ostimes, deaths).item()
    logger.info(f'Concordance-index: {c_index}')
    writer.add_scalar(tag="loss/train", scalar_value=losses.avg, global_step=epoch)
    writer.add_scalar(tag="c_index/train", scalar_value=c_index, global_step=epoch)


def inference(model, valid_dataset, criterion, logger, config, best_perf, epoch, writer):
    model.eval()

    Cox_loss = AverageMeter()
    loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.INFERENCE.BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )
    risks, ostimes, deaths = [], [], []
    for idx, data_dict in enumerate(loader):
        data = data_dict['data']
        data_def = data_dict['data_def']
        death = data_dict['death']
        ostime = data_dict['ostime']
        info = data_dict['info']

        data = data.cuda()
        data_def = data_def.cuda()
        death = death.cuda()
        ostime = ostime.cuda()
        info = info.cuda()
        # run training
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    data_ti = torch.chunk(data, 4, dim=1)
                    data1 = data_ti[0]
                _, _, _, _, risk = model(data1, data_def, info)
                risk = risk.squeeze()
                ostime, indices = torch.sort(ostime)
                death = death[indices]
                risk = risk[indices]
                loss = criterion(risk, ostime, death)
                risks.append(risk)
                ostimes.append(ostime)
                deaths.append(death)
                Cox_loss.update(loss)
    if risks[-1].shape:
        risks = torch.cat(risks)
        ostimes = torch.cat(ostimes)
        deaths = torch.cat(deaths)
    else:
        risks = torch.cat(risks[:-1])
        ostimes = torch.cat(ostimes[:-1])
        deaths = torch.cat(deaths[:-1])
    ostimes, indices = torch.sort(ostimes)
    risks = risks[indices]
    deaths = deaths[indices]
    c_index = concordance_index(risks, ostimes, deaths).item()
    if c_index > best_perf:
        best_perf = c_index

    writer.add_scalar(tag="loss/val", scalar_value=Cox_loss.avg, global_step=epoch)
    writer.add_scalar(tag="c_index/val", scalar_value=c_index, global_step=epoch)

    logger.info('------------- COX LOSS ----------------')
    logger.info(f'Loss mean: {Cox_loss.avg}')
    logger.info('---------------  scores ---------------')
    logger.info(f'Concordance-index: {c_index}')
    logger.info(f'best_perf: {best_perf}')
    logger.info('--------------- ------- ---------------')
    perf = c_index
    return perf


def inference_Brats20(model, valid_dataset, logger, config, best_perf):
    model.eval()

    loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.INFERENCE.BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )
    risks, ostimes, deaths = [], [], []
    for idx, data_dict in enumerate(loader):
        data = data_dict['data']
        data_def = data_dict['data_def']
        death = data_dict['death']
        ostime = data_dict['ostime']
        info = data_dict['info']

        data = data.cuda()
        data_def = data_def.cuda()
        death = death.cuda()
        ostime = ostime.cuda()
        info = info.cuda()
        # run validate
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                data_def = data_def.unsqueeze(1)
                _, _, _, _, risk = model(data, data_def, info)
                risk = risk.squeeze()
                ostime, indices = torch.sort(ostime)
                data_def = data_def[indices]
                death = death[indices]
                risk = risk[indices]
                risks.append(risk)
                ostimes.append(ostime)
                deaths.append(death)
    if risks[-1].shape:
        risks = torch.cat(risks)
        ostimes = torch.cat(ostimes)
        deaths = torch.cat(deaths)
    else:
        risks = torch.cat(risks[:-1])
        ostimes = torch.cat(ostimes[:-1])
        deaths = torch.cat(deaths[:-1])
    ostimes, indices = torch.sort(ostimes)
    risks = risks[indices]
    deaths = deaths[indices]
    c_index = concordance_index(risks, ostimes, deaths).item()
    if c_index > best_perf:
        best_perf = c_index

    logger.info('---------------  scores ---------------')
    logger.info(f'Concordance-index: {c_index}')
    logger.info(f'best_perf: {best_perf}')
    logger.info('--------------- ------- ---------------')
    perf = c_index
    return perf


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a = torch.tensor([3, 5, 6, 9, 10, 21, 22, 23]).to(device)
    b = torch.tensor([4, 7, 6, 2, 3, 9, 20, 30]).to(device)
    c = torch.tensor([0, 0, 1, 0, 0, 1, 1, 0]).to(device)
    criterion = CoxLoss()
    print(criterion(b, a, c))
    print(concordance_index(b, a, c))

