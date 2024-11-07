from argparse import ArgumentParser
import os
from PIL import Image
import json
import cv2
from pycocotools.coco import COCO
import numpy as np
import torch
import torch.distributed as dist
from datetime import timedelta

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
    parser.add_argument('--dataset_root_dir', type=str, )
    parser.add_argument('--dataset_json_path', type=str, )
    parser.add_argument('--dataset_out_dir', type=str, )
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--mode', type=str, default='square', help='tight | square | padding')
    parser.add_argument('--padding_width', type=int, default=40)
    parser.add_argument('--fill', type=str, default='white', help='white | blur | ori | black')

    parser.add_argument('--dist', action='store_true', default=False, help='whether to save cross attention maps')
    parser.add_argument('--backend', type=str, default='nccl')
    args = parser.parse_args()

    dataset_root_dir = args.dataset_root_dir
    dataset_json_path = args.dataset_json_path
    dataset_out_dir = args.dataset_out_dir
    split = args.split
    mode = args.mode
    fill = args.fill
    
    if os.path.exists(dataset_json_path) == False:
        print('{} not existed'.format(dataset_json_path))
        exit()

    # # init distributed
    if args.dist:
        global_rank, local_rank, world_size, device = init_distributed(backend=args.backend)
    else:
        global_rank = 0
        local_rank = 0
        world_size = 1
        device = 'cuda:0'
    print('local rank: {}, global rank: {}, world size: {}, device: {}'.format(local_rank, global_rank, world_size, device))

    # open json
    '''
    # each video 
    # segmentations: each frame
    [
        {
            "video_id": 5,
            "score": 0.9980283379554749,
            "category_id": 7,
            "segmentations": [{
                "size": [, ],
                "counts",
            }, {}, {}]
        }
    ]
    '''

    '''
    "videos": [{"id": 5, "width": 2048, "height": 1024, "file_names": []]
    '''

    with open(dataset_json_path, 'r') as f:
        gt_json = json.load(f)

    coco = COCO(dataset_json_path)

    image_in_dir = os.path.join(dataset_root_dir)

    image_out_dir = os.path.join(dataset_out_dir, '{}2017_{}_{}_crop'.format(split, mode, fill))

    # create out dir
    if os.path.exists(image_out_dir) == False:
        os.makedirs(image_out_dir)

    # out_json = {
    #     "images" : [],
    #     "annotations" : [],
    #     "categories" : gt_json['categories']
    # }

    category_id_to_name = {}

    for category in gt_json['categories']:
        category_id = category['id']
        category_name = category['name'].lower()
        
        category_id_to_name[category_id] = category_name

        # convert json 1 - 1203 to 0 - 1202
        category_dir = os.path.join(image_out_dir, str(category_id - 1))
        if os.path.exists(category_dir) == False:
            os.makedirs(category_dir)

    print('>>> processing annos')
    image_id_to_name = {}
    for image in gt_json['images']:
        image_id_to_name[image['id']] = image['coco_url']

    # each annotation
    for anno in gt_json['annotations']:
        anno_id = anno['id']
        image_id = anno['image_id']

        img_name = image_id_to_name[image_id].replace('http://images.cocodataset.org/', '')
        segmentation = anno['segmentation']
        category_id = anno['category_id']

        if segmentation == None:
            continue

        bbox = anno['bbox']
        area = anno['area']
        mask = mask = coco.annToMask(anno)

        new_area = np.sum(mask).item()
        hor = np.sum(mask, axis=0)
        if np.sum(hor) == 0:
            continue
        hor_idx = np.nonzero(hor)[0]
        x = hor_idx[0]
        width = hor_idx[-1] - x + 1
        vert = np.sum(mask, axis=1)
        if np.sum(vert) == 0:
            continue
        vert_idx = np.nonzero(vert)[0]
        
        y = vert_idx[0]
        height = vert_idx[-1] - y + 1

        # load image
        image = Image.open(os.path.join(image_in_dir, img_name))
        image_width, image_height = image.size
        new_image_id = '{:012d}_{}'.format(image_id, anno_id)
        new_filename = '{}.png'.format(new_image_id)
        image_out_path = os.path.join(image_out_dir, str(category_id - 1), new_filename)

        if fill == 'white':
            print('Fill {}'.format(fill))
            image_np = np.array(image)
            image_np[mask == 0] = 255
            image = Image.fromarray(image_np)
        elif fill == 'blur':
            print('Fill {}'.format(fill))
            image_np = np.array(image)
            image_np_copy = image_np.copy()
            image_blur = cv2.blur(image_np_copy, (10, 10))
            image_np[mask == 0] = image_blur[mask == 0]
            image = Image.fromarray(image_np)
        elif fill == 'ori':
            print('Fill {}'.format(fill))
            pass
        elif fill == 'black':
            print('Fill {}'.format(fill))
            image_np = np.array(image)
            image_np[mask == 0] = 0
            image = Image.fromarray(image_np)
        else:
            print('fill {} not supported'.format(fill))
            exit()

        if mode == 'square':
            center_x = x + width / 2
            center_y = y + height / 2
            square_width = max(width, height)
            half_square_width = square_width / 2

            # padding
            left = center_x - half_square_width
            right = center_x + half_square_width
            top = center_y - half_square_width
            bottom = center_y + half_square_width

            left_padding = 0
            right_padding = 0
            top_padding = 0
            bottom_padding = 0

            if left < 0:
                left_padding = -left
            if right > image_width:
                right_padding = right - image_width
            if top < 0:
                top_padding = -top
            if bottom > image_height:
                bottom_padding = bottom - image_height
            
            new_width = int(image_width + left_padding + right_padding)
            new_height = int(image_height + top_padding + bottom_padding)

            padding_image = Image.new(image.mode, (new_width, new_height), (0, 0, 0))

            padding_image.paste(image, (int(left_padding), int(top_padding)))

            image = padding_image.crop((int(left_padding + left), int(top_padding + top), int(left_padding + left + square_width), int(top_padding + top + square_width)))

            # new_bbox = [int(left_padding + left), int(top_padding + top), int(square_width), int(square_width)]
        elif mode == 'padding':
            center_x = x + width / 2
            center_y = y + height / 2
            square_width = args.padding_width * 2
            half_square_width = args.padding_width

            # padding
            left = max(x - half_square_width, 0)
            right = min(x + width + half_square_width, image_width)
            top = max(y - half_square_width, 0)
            bottom = min(y + height + half_square_width, image_height)

            image = image.crop((left, top, right, bottom))
        elif mode == 'tight':
            # crop image
            image = image.crop((x, y, x + width, y + height))
            # new_bbox = [0, 0, int(width), int(height)]
        else:
            print('mode not matched')
            exit()

        # save image
        image.save(image_out_path)
        print('save {}'.format(new_filename))

    print('done')

