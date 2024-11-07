import os
import argparse
import json
import shutil
from datetime import timedelta
from glob import glob

import torch
import torch.distributed as dist
import numpy as np

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
    parser = argparse.ArgumentParser(description='Process input args.')
    parser.add_argument(
        '--indir',
        type=str,
        help='if specified, load prompts from this file',
        action='append'
        # default='lvis_v1_val_1.txt'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        nargs='?',
        help='dir to write results to',
        default='output/txt2img-samples'
    )
    parser.add_argument(
        '--dist',
        action='store_true',
        default=False,
        help='whether to save cross attention maps',
    )
    parser.add_argument(
        '--backend',
        type=str, 
        default='nccl'
    )
    parser.add_argument(
        '--overwrite', 
        action='store_true', 
        default=False
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=1,
        help='how many samples to produce for each given prompt. A.k.a. batch size',
    )
    parser.add_argument(
        '--in_lvis_json_path', 
        type=str, 
    )
    parser.add_argument(
        '--stages',
        type=str,
        nargs='+',
        default="'I' 'II'",
        help='model stages to use',
    )

    args = parser.parse_args()

    # # init distributed
    if args.dist:
        global_rank, local_rank, world_size, device = init_distributed(backend=args.backend)
    else:
        global_rank = 0
        local_rank = 0
        world_size = 1
        device = 'cuda:0'
    print('local rank: {}, global rank: {}, world size: {}, device: {}'.format(local_rank, global_rank, world_size, device))

    with open(args.in_lvis_json_path, 'r') as f:
        data = json.load(f)

    id_to_name = {d['id']: d['name'] for d in data}

    # create output dir
    if global_rank == 0:
        for stage in args.stages:
            for category_id, category_name in id_to_name.items():
                outdir = os.path.join(args.outdir, stage, category_name)
                if os.path.exists(outdir) == False:
                    os.makedirs(outdir)
                    print('>>> Create dir: {}'.format(outdir))
    
    if args.dist:
        dist.barrier()
        
    for stage in args.stages:
        picked_image_paths = []
        for current_in_dir in args.indir:
            sample_dir = os.path.join(current_in_dir, 'samples', stage)

            stage_sample_paths = sorted(glob(os.path.join(sample_dir, '*.png')))

            for i, stage_sample_path in enumerate(stage_sample_paths):
                if i % world_size == global_rank:
                    picked_image_paths.append(stage_sample_path)
            
        for i, picked_image_path in enumerate(picked_image_paths):
            filename = os.path.basename(picked_image_path)
            category_id = int(filename.split('_')[0])

            category_name = id_to_name[category_id]

            outpath = os.path.join(args.outdir, stage, category_name, filename)

            if os.path.exists(outpath) and args.overwrite == False:
                print('>>> {} is existed'.format(outpath))
                continue

            shutil.copyfile(picked_image_path, outpath)
            print('>>> Copy {} to {}'.format(picked_image_path, outpath))
    
    if args.dist:
        dist.barrier()

    if global_rank == 0:
        for stage in args.stages:
            for category_id, category_name in id_to_name.items():
                len_samples = len(glob(os.path.join(args.outdir, stage, category_name, '*.png')))
                if len_samples != args.n_samples:
                    print('>>> {} only has {} images, but expected to have {}'.format(category_name, len_samples, args.n_samples))
    
    print('done')