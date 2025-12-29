import argparse
import math
import os, sys
import random
import time
import json
import numpy as np
import datetime

import torch
import torch.optim
import torch.utils.data
from torch.cuda.amp import autocast
import torch.distributed as dist

import cv2

from src_files.utils.logger import setup_logger
from src_files.utils.meter import AverageMeter, ProgressMeter
from src_files.utils.helper import clean_state_dict, sl_mAP_cf1_of1
from src_files.models.factory import create_model
from src_files.data.data import get_datasets

from torch.cuda.amp import autocast

import matplotlib.pyplot as plt

NUM_CLASS = {'voc2007': 20, 'coco2014': 80,  'vg256':256, 'Multi-label-dataset': 5}

def get_args():
    parser = argparse.ArgumentParser(description='Clean ASL Training')

    # data
    parser.add_argument('--data_name', help='dataset name', default='coco2014', choices=['voc2007', 'coco2014', 'vg256', 'Multi-label-dataset'])
    parser.add_argument('--data_dir', help='dir of all datasets', default='/home/worker/3DReconstruction2/classification/MLC-base/dataset')
    parser.add_argument('--image_size', default=384, type=int,
                        help='size of input images')
    parser.add_argument('--output', metavar='DIR', default='./outputs',
                        help='path to output folder')

    # model
    parser.add_argument('--model_name', default='q2l_dinov3_vitl16')
    parser.add_argument('--pretrain_type', default='in21k', choices=['in1k', 'in21k','oi'])
    parser.add_argument('--pretrain_dir', default='/home/lthpc/student/zby_weather/Multi-label-exp/MLC-base/pretrained', type=str)

    # train
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        help='batch size')
    parser.add_argument('-p', '--print_freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')

    # distribution training
    parser.add_argument('--distributed', action='store_true', help='using dataparallel')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    # parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--clip_model_name', default="ViT-B/32", type=str)
    parser.add_argument('--mask_prob', default=0.2, type=int)

    parser.add_argument('--n_grid', default=2, type=int)
    parser.add_argument('--logits_attention', default='cross', type=str)

    # random seed
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')

    # resume
    parser.add_argument('--resume', help='dir of all datasets'
                        ,
                        default='/home/lthpc/student/zby_weather/Multi-label-exp/Paper/4-DINOV3+clip/dinov3/outputs/Multi-label-dataset/cross_q2l_dinov3_vitl16_pretrain_dinov3_vitl16_oi_224_adamw_0.0001_0.0001_4_0.05_16_80_seed_1/model_best.pth.tar')

    args = parser.parse_args()
    # 设置设备
    local_rank = int(os.environ['LOCAL_RANK'])
    args.local_rank = local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    args.num_classes = NUM_CLASS[args.data_name]
    args.data_dir = os.path.join(args.data_dir, args.data_name)

    args.output = os.path.join(args.output, args.data_name,
                               f'inference_dist_{args.model_name}_{args.pretrain_type}_{args.image_size}_{args.batch_size}')
    return args


def main():
    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    os.makedirs(args.output, exist_ok=True)

    # setup dist training
    args.distributed = args.distributed or ('WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1)
    
    if args.distributed:
        if not dist.is_initialized():
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.device = 'cuda:%d' % args.local_rank
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        print(f'Training in distributed mode with multiple processes, 1 GPU per process. Process {args.rank}, total {args.world_size}.')
    else:
        args.device = 'cuda:0'
        args.world_size = 1
        args.rank = 0
        print('Training with a single process on 1 GPU.')
    assert args.rank >= 0

    logger = setup_logger(output=args.output, distributed_rank=args.rank, color=False, name="Coco")
    logger.info("Command: " + ' '.join(sys.argv))
    if not args.distributed or dist.get_rank() == 0:
        path = os.path.join(args.output, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))
        os.makedirs(os.path.join(args.output, 'tmpdata'), exist_ok=True)

    return main_worker(args, logger)


pos_ratio = 0.0


def main_worker(args, logger):
    # build model
    global pos_ratio

    logger.info('creating model...')
    model = create_model(args).cuda()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    # Data loading code
    train_dataset, val_dataset = get_datasets(args, patch=True)
    train_labels = train_dataset.Y
    if not args.distributed or dist.get_rank() == 0:
        pos_ratio = train_labels.sum(0) / train_labels.shape[0]
        # print(pos_ratio)

    logger.info("len(val_dataset)): {}".format(len(val_dataset)))

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        batch_size = 128 // args.world_size
        assert args.batch_size % args.world_size == 0, 'Batch size is not divisible by num of gpus.'
    else:
        val_sampler = None
        batch_size = 128

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler)

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(args.local_rank if args.distributed else 0))
            state_dict = clean_state_dict(checkpoint['state_dict_ema'])
            torch.save
            logger.info(checkpoint['best_mAP'])
            model.module.load_state_dict(state_dict, strict=True)
            del checkpoint
            del state_dict
            torch.cuda.empty_cache()
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    mAP_ori, APs_ori, mAP_pat, APs_pat, mAP_mix, APs_mix, outputs = patch_validate(val_loader, model, args, logger)

    return 0


def calculate_metric(preds, labels):
    n_correct_pos = (labels * preds).sum(0)
    n_pred_pos = ((preds == 1)).sum(0)
    n_true_pos = labels.sum(0)
    OP = n_correct_pos.sum() / n_pred_pos.sum()
    CP = np.nanmean(n_correct_pos / n_pred_pos)
    OR = n_correct_pos.sum() / n_true_pos.sum()
    CR = np.nanmean(n_correct_pos / n_true_pos)

    CF1 = (2 * CP * CR) / (CP + CR)
    OF1 = (2 * OP * OR) / (OP + OR)

    return CP, CR, CF1, OP, OR, OF1


def thresholding(labels, probs, logger):
    for thre in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        preds = (probs > thre).astype(np.float32)
        metrics = calculate_metric(preds, labels)
        logger.info(f'{thre}, {np.round(np.array(metrics) * 100, decimals=1)}')

    return 0


def weighted_sum(batch_size, logits_pat_1, logits_pat_2):
    split_list1 = torch.split(logits_pat_1, batch_size)  # [64,80] -> 4 * [16,80]
    logits_joint1 = torch.stack(split_list1, dim=1)  # 4 * [16,80] -> [16, 4, 80]
    logits_sfmx1 = torch.softmax(logits_joint1, dim=1)  # [16, {4}, 80]

    split_list2 = torch.split(logits_pat_2, batch_size)  # [64,80] -> 4 * [16,80]
    logits_joint2 = torch.stack(split_list2, dim=1)  # 4 * [16,80] -> [16, 4, 80]

    logits_joint = (logits_sfmx1 * logits_joint2).sum(dim=1)  # [16, 4, 80] -> [16,80]
    return logits_joint

@torch.no_grad()
def patch_validate(val_loader, model, args, logger):
    batch_time = AverageMeter('Time', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, mem],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    probs_ori = []
    probs_pat = []
    probs_mix = []
    labels = []

    end = time.time()
    for i, (inputs, targets) in enumerate(val_loader):

        batch_size = targets.shape[0]
        inputs = torch.cat(inputs, dim=0).cuda(non_blocking=True)

        # compute output
        with autocast():
            outputs = model(inputs)

        outputs_ori = outputs[0][:batch_size]

        if args.logits_attention == 'self':
            outputs_pat_1, outputs_pat_2 = outputs[1][batch_size:], outputs[1][batch_size:]
        elif args.logits_attention == 'cross':
            outputs_pat_1, outputs_pat_2 = outputs[1][batch_size:], outputs[2][batch_size:]
        else:
            print("attention: {} not found !!".format(args.logits_attention))
            exit(-1)

        outputs_pat = weighted_sum(batch_size, outputs_pat_1, outputs_pat_2)
        outputs_mix = (outputs_ori + outputs_pat) / 2

        # add list
        probs_ori.append(torch.sigmoid(outputs_ori).detach().cpu())
        probs_pat.append(torch.sigmoid(outputs_pat).detach().cpu())
        probs_mix.append(torch.sigmoid(outputs_mix).detach().cpu())
        labels.append(targets.detach().cpu())

        # record memory
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % (2 * args.print_freq) == 0:
            progress.display(i, logger)

    # saved data
    labels = torch.cat(labels).numpy()
    probs_ori = torch.cat(probs_ori).numpy()
    probs_pat = torch.cat(probs_pat).numpy()
    probs_mix = torch.cat(probs_mix).numpy()

    data_ori = np.concatenate((probs_ori, labels), axis=1)
    saved_name_ori = 'tmpdata/data_ori_tmp.{}.txt'.format(args.rank if args.distributed else 0)
    np.savetxt(os.path.join(args.output, saved_name_ori), data_ori)
    data_pat = np.concatenate((probs_pat, labels), axis=1)
    saved_name_pat = 'tmpdata/data_pat_tmp.{}.txt'.format(args.rank if args.distributed else 0)
    np.savetxt(os.path.join(args.output, saved_name_pat), data_pat)
    data_mix = np.concatenate((probs_mix, labels), axis=1)
    saved_name_mix = 'tmpdata/data_mix_tmp.{}.txt'.format(args.rank if args.distributed else 0)
    np.savetxt(os.path.join(args.output, saved_name_mix), data_mix)

    if dist.get_world_size() > 1:
        dist.barrier()

    if dist.get_rank() == 0:
        global pos_ratio

        logger.info("Calculating mAP:")

        # 原始数据
        filenamelist_ori = ['tmpdata/data_ori_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
        mAP_ori, APs_ori, all_list_ori, top3_list_ori, outputs_ori, precision_ori, recall_ori, top_precision_ori, top_recall_ori = sl_mAP_cf1_of1(
            [os.path.join(args.output, _filename) for _filename in filenamelist_ori], args.num_classes, pos_ratio)

        # 病变数据
        filenamelist_pat = ['tmpdata/data_pat_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
        mAP_pat, APs_pat, all_list_pat, top3_list_pat, outputs_pat, precision_pat, recall_pat, top_precision_pat, top_recall_pat = sl_mAP_cf1_of1(
            [os.path.join(args.output, _filename) for _filename in filenamelist_pat], args.num_classes, pos_ratio)

        # 混合数据
        filenamelist_mix = ['tmpdata/data_mix_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
        mAP_mix, APs_mix, all_list_mix, top3_list_mix, outputs_mix, precision_mix, recall_mix, top_precision_mix, top_recall_mix = sl_mAP_cf1_of1(
            [os.path.join(args.output, _filename) for _filename in filenamelist_mix], args.num_classes, pos_ratio)

        logger.info("#####################################################################")

        # 输出 mAP 和其他指标
        logger.info("mAP ori {:.2f}".format(mAP_ori))
        logger.info("CP {:.2f}, CR {:.2f}, CF1 {:.2f}, OP {:.2f}, OR {:.2f}, OF1 {:.2f}".format(all_list_ori[0],
                                                                                                all_list_ori[1],
                                                                                                all_list_ori[2],
                                                                                                all_list_ori[3],
                                                                                                all_list_ori[4],
                                                                                                all_list_ori[5]))
        logger.info("Top-3 CP {:.2f}, CR {:.2f}, CF1 {:.2f}, OP {:.2f}, OR {:.2f}, OF1 {:.2f}".format(top3_list_ori[0],
                                                                                                      top3_list_ori[1],
                                                                                                      top3_list_ori[2],
                                                                                                      top3_list_ori[3],
                                                                                                      top3_list_ori[4],
                                                                                                      top3_list_ori[5]))

        # 输出每个类别的 Precision 和 Recall
        logger.info("Precision ori: {}".format(precision_ori))
        logger.info("Recall ori: {}".format(recall_ori))
        logger.info("Top-3 Precision ori: {}".format(top_precision_ori))
        logger.info("Top-3 Recall ori: {}".format(top_recall_ori))

        logger.info("#####################################################################")

        # 输出病变数据
        logger.info("mAP pat {:.2f}".format(mAP_pat))
        logger.info("CP {:.2f}, CR {:.2f}, CF1 {:.2f}, OP {:.2f}, OR {:.2f}, OF1 {:.2f}".format(all_list_pat[0],
                                                                                                all_list_pat[1],
                                                                                                all_list_pat[2],
                                                                                                all_list_pat[3],
                                                                                                all_list_pat[4],
                                                                                                all_list_pat[5]))
        logger.info("Top-3 CP {:.2f}, CR {:.2f}, CF1 {:.2f}, OP {:.2f}, OR {:.2f}, OF1 {:.2f}".format(top3_list_pat[0],
                                                                                                      top3_list_pat[1],
                                                                                                      top3_list_pat[2],
                                                                                                      top3_list_pat[3],
                                                                                                      top3_list_pat[4],
                                                                                                      top3_list_pat[5]))

        # 输出每个类别的 Precision 和 Recall
        logger.info("Precision pat: {}".format(precision_pat))
        logger.info("Recall pat: {}".format(recall_pat))
        logger.info("Top-3 Precision pat: {}".format(top_precision_pat))
        logger.info("Top-3 Recall pat: {}".format(top_recall_pat))

        logger.info("#####################################################################")

        # 输出混合数据
        logger.info("mAP mix {:.2f}".format(mAP_mix))
        logger.info("CP {:.2f}, CR {:.2f}, CF1 {:.2f}, OP {:.2f}, OR {:.2f}, OF1 {:.2f}".format(all_list_mix[0],
                                                                                                all_list_mix[1],
                                                                                                all_list_mix[2],
                                                                                                all_list_mix[3],
                                                                                                all_list_mix[4],
                                                                                                all_list_mix[5]))
        logger.info("Top-3 CP {:.2f}, CR {:.2f}, CF1 {:.2f}, OP {:.2f}, OR {:.2f}, OF1 {:.2f}".format(top3_list_mix[0],
                                                                                                      top3_list_mix[1],
                                                                                                      top3_list_mix[2],
                                                                                                      top3_list_mix[3],
                                                                                                      top3_list_mix[4],
                                                                                                      top3_list_mix[5]))

        # 输出每个类别的 Precision 和 Recall
        logger.info("Precision mix: {}".format(precision_mix))
        logger.info("Recall mix: {}".format(recall_mix))
        logger.info("Top-3 Precision mix: {}".format(top_precision_mix))
        logger.info("Top-3 Recall mix: {}".format(top_recall_mix))

        logger.info("#####################################################################")

        # 保存图表等结果
        plt.figure(figsize=(16, 8))
        bar_width = 0.25
        index = np.arange(len(APs_ori))
        plt.bar(index, APs_ori, bar_width, label='Ori')
        plt.bar(index + bar_width, APs_pat, bar_width, label='Pat')
        plt.bar(index + 2 * bar_width, APs_mix, bar_width, label='Mix')
        plt.savefig(os.path.join(args.output, 'bar.jpg'), dpi=500)

        # 保存输出
        print(outputs_mix[0].shape, outputs_mix[1].shape, outputs_mix[2].shape)
        print(type(outputs_mix[0]), type(outputs_mix[1]), type(outputs_mix[2]))
        np.save(os.path.join(args.output, 'probs.npy'), outputs_mix[0])
        np.save(os.path.join(args.output, 'preds.npy'), outputs_mix[1])
        np.save(os.path.join(args.output, 'labels.npy'), outputs_mix[2])

        thresholding(outputs_mix[2], outputs_mix[0], logger)

    else:
        mAP_ori = 0
        APs_ori = []
        mAP_pat = 0
        APs_pat = []
        mAP_mix = 0
        APs_mix = []

    return mAP_ori, APs_ori, mAP_pat, APs_pat, mAP_mix, APs_mix, outputs_mix


if __name__ == '__main__':
    main()