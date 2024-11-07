from argparse import ArgumentParser
import os
from PIL import Image
import clip
import json
import cv2
from pycocotools.coco import COCO
import numpy as np
import torch
import torch.distributed as dist
from datetime import timedelta, datetime
from glob import glob
import csv
import matplotlib.pyplot as plt

def init_distributed(backend='nccl'):
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # If the OS is Windows or macOS, use gloo instead of nccl
    if world_size > 0:
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=720000))

    # set distributed device
    device = torch.device('cuda:{}'.format(local_rank))
    return rank, local_rank, world_size, device

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_model_path', type=str)
    parser.add_argument('--output_model_path', type=str)
    parser.add_argument('--input_num_category', type=int)
    parser.add_argument('--output_num_category', type=int)

    parser.add_argument('--dist', action='store_true', default=False)
    parser.add_argument('--backend', type=str, default='nccl')
    args = parser.parse_args()

    # init distributed
    if args.dist:
        global_rank, local_rank, world_size, device = init_distributed(backend=args.backend)
    else:
        global_rank = 0
        local_rank = 0
        world_size = 1
        device = 'cuda:0'
    print('local rank: {}, global rank: {}, world size: {}, device: {}'.format(local_rank, global_rank, world_size, device))

    args.device = device

    # print start datetime
    start_time = datetime.now()
    print('>>> Start datetime: {}'.format(str(start_time)))

    # load model
    print('>>> Load model from {}'.format(args.input_model_path))
    state_dict = torch.load(args.input_model_path)

    print('>>> Convert model from {} to {}'.format(args.input_num_category, args.output_num_category))

    print('>>> Process model')
    model = state_dict['model']
    for key in list(model.keys()):
        if 'roi_heads.box_predictor' in key:
            if 'cls_score' in key:
                if 'bias' in key:
                    assert model[key].shape[0] == (args.input_num_category + 1)
                    print('>>> Replace {}'.format(key))
                    # (args.input_num_category + 1) -> args.output_num_category
                    temp_weight = model[key]
                    # 取前args.output_num_category个类，以及最后的背景类
                    model[key] = torch.cat((temp_weight[:args.output_num_category], temp_weight[-1:]), dim=0)
                elif 'weight' in key:
                    assert model[key].shape[0] == (args.input_num_category + 1)
                    print('>>> Replace {}'.format(key))
                    # (args.input_num_category + 1) -> args.output_num_category
                    temp_weight = model[key]
                    # 取前args.output_num_category个类，以及最后的背景类
                    model[key] = torch.cat((temp_weight[:args.output_num_category], temp_weight[-1:]), dim=0)
            elif 'freq_weight' in key:
                assert model[key].shape[0] == args.input_num_category
                print('>>> Replace {}'.format(key))
                # args.input_num_category -> args.output_num_category
                temp_weight = model[key]
                # 取前args.output_num_category个类，以及最后的背景类
                model[key] = temp_weight[:args.output_num_category]
    
    print('>>> Process model_ema')
    model_ema = state_dict['model_ema']
    for key in list(model_ema.keys()):
        if 'roi_heads.box_predictor' in key:
            if 'cls_score' in key:
                if 'bias' in key:
                    assert model_ema[key].shape[0] == (args.input_num_category + 1)
                    print('>>> Replace {}'.format(key))
                    # (args.input_num_category + 1) -> args.output_num_category
                    temp_weight = model_ema[key]
                    # 取前args.output_num_category个类，以及最后的背景类
                    model_ema[key] = torch.cat((temp_weight[:args.output_num_category], temp_weight[-1:]), dim=0)
                elif 'weight' in key:
                    assert model_ema[key].shape[0] == (args.input_num_category + 1)
                    print('>>> Replace {}'.format(key))
                    # (args.input_num_category + 1) -> args.output_num_category
                    temp_weight = model_ema[key]
                    # 取前args.output_num_category个类，以及最后的背景类
                    model_ema[key] = torch.cat((temp_weight[:args.output_num_category], temp_weight[-1:]), dim=0)
            elif 'freq_weight' in key:
                assert model_ema[key].shape[0] == args.input_num_category
                print('>>> Replace {}'.format(key))
                # args.input_num_category -> args.output_num_category
                temp_weight = model_ema[key]
                # 取前args.output_num_category个类，以及最后的背景类
                model_ema[key] = temp_weight[:args.output_num_category]
    
    dirname = os.path.dirname(args.output_model_path)
    if not os.path.exists(dirname):
        print('>>> Create {}'.format(dirname))
        os.makedirs(dirname)
    
    print('>>> Save model to {}'.format(args.output_model_path))
    torch.save(state_dict, args.output_model_path)

    end_time = datetime.now()
    print('>>> End datetime: {}, used {}'.format(str(end_time), str(end_time - start_time)))