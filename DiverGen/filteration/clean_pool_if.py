import argparse
import shutil
import cv2
import json
import multiprocessing as mp
import numpy as np
import os
import csv
import random
import torch
import torch.distributed as dist
from PIL import Image, ImageFile
from collections import defaultdict
from datetime import timedelta

import tempfile

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

def filter_none(x):
    return [i for i in x if i is not None]

def get_largest_connect_component(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    if len(area) >= 1:
        max_idx = np.argmax(area)
        img2 = np.zeros_like(img)
        cv2.fillPoly(img2, [contours[max_idx]], 1)
        return img2
    else:
        return img


def subwork(img_path, output):
    mask_path = None
    if '|' in img_path:
        mask_path = img_path.split('|')[1]
        img_path = img_path.split('|')[0]
    try:
        img_RGBA = np.array(Image.open(img_path).convert('RGBA'))
    except:
        return None
    if mask_path is not None:
        try:
            img_RGBA[:, :, -1] = np.array(Image.open(mask_path))
        except:
            return None
    alpha = img_RGBA[..., 3:]
    seg_mask = (alpha > 128).astype('uint8')
    seg_mask = get_largest_connect_component(seg_mask)
    if seg_mask.size == 0:
        return None

    seg_mask_ = np.where(seg_mask)
    if seg_mask_[0].size == 0 \
        or seg_mask_[1].size == 0:
        return None
    
    try:
        y_min, y_max, x_min, x_max = np.min(seg_mask_[0]), np.max(seg_mask_[0]), np.min(seg_mask_[1]), np.max(seg_mask_[1])
    except Exception as e:
        print(e)
        return None
    
    if y_max <= y_min or x_max <= x_min:
        return None
    img_RGBA[:, :, 3:] *= seg_mask
    img_RGBA = img_RGBA[y_min:y_max + 1, x_min:x_max + 1]
    pil_image = Image.fromarray(img_RGBA)
    pil_image.save(output)
    return '*' + output


def work(part):
    output_path = part['output']
    del part['output']
    for i in part:
        print(i)
        os.makedirs(os.path.join(output_path, 'images', str(i)), exist_ok=True)
    res = {i: filter_none(
        [subwork(j, os.path.join(output_path, 'images', str(i), '{}.png'.format(c))) for c, j in enumerate(part[i])])
           for i in part}
    print('done', {i: len(res[i]) for i in res})
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--filter_image_csv_path', type=str, default=None)
    parser.add_argument('--min_clip', type=float, default=25)
    parser.add_argument('--min_area', type=float, default=0.0)
    parser.add_argument('--max_area', type=float, default=1.0)
    parser.add_argument('--tolerance', type=float, default=1)
    parser.add_argument('--dist', action='store_true', default=False)
    parser.add_argument('--seg_method', type=str, nargs='+')
    parser.add_argument('--stages', type=str, nargs='+', default="I II", help='model stages to use')
    
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

    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    seg_methods = args.seg_method

    if args.filter_image_csv_path is not None:
        filter_image_paths = {}
        print('>>> Load filter image csv file: {}'.format(args.filter_image_csv_path))
        # open csv file
        with open(args.filter_image_csv_path, 'r') as f:
            # read csv file
            reader = csv.reader(f)
            # get the first column as filename, and last column as similarity
            # skip header
            next(reader)
            for row in reader:
                category_name = row[0]
                filename = row[1]

                if category_name not in filter_image_paths:
                    filter_image_paths[category_name] = set()
                
                filter_image_paths[category_name].add(filename)       

    for stage in args.stages:
        results = []
        for seg_method in seg_methods:
            with open(os.path.join(args.input_dir, stage, seg_method, 'results.json')) as f:
                data = json.load(f)
            data = sorted(data, key=lambda x: x['image_count'])
            results.append(data)
    
        count = 0
        datadict = defaultdict(list)
    
        for c in zip(*results):
            ids = [np.array(j['id']) for j in c]
            assert ids.count(ids[0]) == len(ids), 'id not match, {}'.format(ids)
            npc = np.stack([np.array(j['clip_scores']) for j in c], 0)
            areas = np.stack([np.array(j['areas']) for j in c], 0)
            if npc.size == 0 or areas.size == 0:
                continue

            
            name = c[0]['name']
            cid = c[0]['id'] - 1
            npx = np.argmax(npc, 0)
            this_bar = min(args.min_clip, np.max(npc) - args.tolerance)

            if args.enable_split:
                start_index = args.start_index
                end_index = args.end_index
            else:
                start_index = 0
                end_index = len(npx)

            print('start index: {}, end index: {}'.format(start_index, end_index))

            for k in range(start_index, end_index):
                current_filename = f"{c[0]['id']}_{k:07d}"
                current_image_path = os.path.join(args.image_dir, stage, name, f"{current_filename}.png")

                if args.filter_image_csv_path is not None:
                    if name not in filter_image_paths:
                        print('category similarity is too low, skip {}'.format(current_image_path))
                        continue
                    if f"{current_filename}.png" not in filter_image_paths[name]:
                        print('similarity is too low, skip {}'.format(current_image_path))
                        
                        # copy mask file to log dir
                        # shutil.copy(current_image_path, os.path.join(args.output_log_dir, 'similarity_smaller', f"{current_filename}.png"))

                        continue

                if npc[npx[k], k] < this_bar or areas[npx[k], k] < args.min_area or areas[npx[k], k] > args.max_area:
                    print('area is too small or too large, skip {}'.format(current_image_path))
                    continue

                seg_method = seg_methods[npx[k]]
                print('{} is selected'.format(seg_method))
                current_mask_path = os.path.join(args.input_dir, stage, seg_method, name, f"{c[0]['id']}_{k:07d}.png")
                # copy mask file to log dir
                # shutil.copy(seg_mask_path, os.path.join(args.output_log_dir, 'save', f"{current_filename}.png"))
                datadict[cid].append('|'.join([current_image_path, current_mask_path]))
                count += 1
    
        output_path = os.path.dirname(args.output_file)
        mp.set_start_method('spawn', force=True)
        num_threads = 128
        pool = mp.Pool(processes=num_threads)
        '''
        parts = [
            {
                category_id: [
                    category_image_path_1|category_mask_path_1,
                    category_image_path_2|category_mask_path_2,
                    ...
                ],
                output: output_json_dir
            },
            ...
        ]
        '''
        parts = [{i: datadict[i], 'output': output_path} for i in datadict]

        print('>>> Start processing')
        results = pool.map(work, parts, 1)
        result = {}
        for i in results:
            result.update(i)
        
        print('>>> Save json to {}'.format(args.output_file))
        with open(args.output_file, 'w') as f:
            json.dump(result, f)
