import os
import shutil
import random
import numpy as np
import json
from argparse import ArgumentParser
from PIL import Image

from segment_anything import build_sam, SamPredictor 

from datetime import timedelta
import torch
import torch.distributed as dist

def init_distributed(backend='nccl'):
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # If the OS is Windows or macOS, use gloo instead of nccl
    if world_size > 0:
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=7200))

    # set distributed device
    device = torch.device('cuda:{}'.format(local_rank))
    return rank, local_rank, world_size, device

def check_point_in_foreground(coord, atten_map, threshold):
    x, y = coord
    return atten_map[x, y] > threshold

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in_img_dir', type=str, )
    parser.add_argument('--out_mask_dir', type=str, )
    parser.add_argument('--in_lvis_json_path', type=str, )
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--stages', type=str, nargs='+', default="I II", help='model stages to use')
    parser.add_argument('--seg_name', type=str, )
    parser.add_argument('--pesudo_world_size', type=int, default=None)
    parser.add_argument('--pesudo_global_rank', type=int, default=None)
    parser.add_argument('--sam_checkpoint_path', type=str, )
    parser.add_argument('--background_mode', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--corner_margin', type=int, default=5)
    parser.add_argument('--corner_location', type=str, nargs='+')
    parser.add_argument('--dist', action='store_true', default=False)
    parser.add_argument('--n_samples', type=int, default=1, help='how many samples to produce for each given prompt. A.k.a. batch size')

    args = parser.parse_args()

    # init distributed
    if args.dist:
        global_rank, local_rank, world_size, device = init_distributed()
    else:
        global_rank = 0
        local_rank = 0
        world_size = 1
        device = 'cuda:0'
    print('local rank: {}, global rank: {}, world size: {}, device: {}'.format(local_rank, global_rank, world_size, device))

    torch.cuda.set_device(device)

    in_npy_dir = args.in_npy_dir
    out_mask_dir = args.out_mask_dir

    sam = build_sam(checkpoint=args.sam_checkpoint_path)
    sam.to(device)
    predictor = SamPredictor(sam)

    with open(args.in_lvis_json_path, 'r') as f:
        data = json.load(f)

    # sort data by data image_count from small to large
    if 'image_count' in data[0]:
        data = sorted(data, key=lambda x: x['image_count'])
    else:
        data = sorted(data, key=lambda x: x['id'])

    if global_rank == 0:
        for stage in args.stages:
            for category in data:
                category_name = category['name']
                if stage == 'sd':
                    outdir = os.path.join(out_mask_dir, args.seg_name, category_name)
                else:
                    outdir = os.path.join(out_mask_dir, stage, args.seg_name, category_name)
                if os.path.exists(outdir) == False:
                    os.makedirs(outdir)
                    print('>>> Create dir: {}'.format(outdir))
    else:
        print('==> Waiting for creating dir in rank {}...'.format(global_rank))
    
    if args.dist:
        dist.barrier()
    
    for stage in args.stages:
        if stage == 'sd':
            current_in_npy_dir = args.in_npy_dir
            current_in_img_dir = args.in_img_dir
            current_out_mask_dir = out_mask_dir
        else:
            current_in_npy_dir = os.path.join(args.in_npy_dir, stage)
            current_in_img_dir = os.path.join(args.in_img_dir, stage)
            current_out_mask_dir = os.path.join(out_mask_dir, stage)

        for category in data:
            category_name = category['name']
            sample_dir = os.path.join(current_in_img_dir, category_name)
            mask_dir = os.path.join(current_out_mask_dir, args.seg_name, category_name)

            if not os.path.exists(sample_dir):
                print('>>> Skip {}, it does not exist'.format(category_name))
                continue

            sample_filenames = sorted(os.listdir(sample_dir))

            len_samples = len(sample_filenames)
            # if len_samples != args.n_samples:
            #     print('>>> Skip {}, it has {} images, but expected to have {}'.format(category_name, len_samples, args.n_samples))
            #     continue

            # if global_rank == 0:
            #     if not os.path.exists(mask_dir):
            #         os.makedirs(mask_dir)

            picked_filenames = []

            if args.pesudo_world_size is not None and args.pesudo_global_rank is not None:
                for i, stage_sample_filename in enumerate(sample_filenames):
                    if i % args.pesudo_world_size == args.pesudo_global_rank:
                        picked_filenames.append(stage_sample_filename)
            else:
                for i, stage_sample_filename in enumerate(sample_filenames):
                    if i % world_size == global_rank:
                        picked_filenames.append(stage_sample_filename)
    
            for filename in picked_filenames:
                print('processing {}'.format(filename))
                in_img_path = os.path.join(sample_dir, filename)
                out_mask_path = os.path.join(mask_dir, filename)
    
                if args.overwrite == False and os.path.exists(out_mask_path):
                    print('skip {}'.format(out_mask_path))
                    continue

                img = Image.open(in_img_path)
                img = np.array(img)
        
                if args.background_mode:
                    corner_locations = []
                    corner_labels = []
                    corner_locations = [
                        [args.corner_margin, args.corner_margin],
                        [0, img.shape[1] - 1 - args.corner_margin],
                        [img.shape[0] - 1 - args.corner_margin, args.corner_margin],
                        [img.shape[0] - 1 - args.corner_margin, img.shape[1] - 1 - args.corner_margin]]
                    corner_labels = [1, 1, 1, 1]
                    
                    coords = np.array(corner_locations)
                    coord_labels = np.array(corner_labels)
    
                predictor.reset_image()
                predictor.set_image(img)
    
                # points
                masks, _, _ = predictor.predict(point_coords = coords, point_labels = coord_labels)
                
                mask = masks[2]
                mask = mask.astype(np.uint8)
                mask = 1 - mask
                mask *= 255
                img_mask = Image.fromarray(mask)
                img_mask.save(out_mask_path)