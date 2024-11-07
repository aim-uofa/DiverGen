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

class ImageSimilarity():
    def __init__(self, method = 'clip', args = None):
        self.method = method
        self.args = args
        if self.method == 'clip' and args.disable_init == False:
            # load clip
            print('>>> Loading CLIP...')
            if args.clip_ckpt_dir is None:
                self.clip_model, self.preprocess = clip.load("ViT-L/14", device=args.device)
            else:
                self.clip_model, self.preprocess = clip.load("ViT-L/14", device=args.device, download_root=args.clip_ckpt_dir)
            self.clip_model.float()
    
    def get_similarity(self, image1, image2):
        if self.method == 'clip':
            with torch.no_grad():
                # encode by clip
                image1 = self.preprocess(image1).unsqueeze(0).to(self.args.device)
                image1_features = self.clip_model.encode_image(image1)
    
                # encode by clip
                image2 = self.preprocess(image2).unsqueeze(0).to(self.args.device)
                image2_features = self.clip_model.encode_image(image2)
    
                similarity = torch.cosine_similarity(image1_features, image2_features).cpu().item()

            return similarity
        else:
            raise NotImplementedError
        
    def get_similarity_batch(self, image1, image2_list):
        if self.method == 'clip':
            with torch.no_grad():
                # encode by clip
                image1 = self.preprocess(image1).unsqueeze(0).to(self.args.device)
                image1_features = self.clip_model.encode_image(image1)
    
                # encode by clip
                image2_list = [self.preprocess(image2) for image2 in image2_list]
                image2 = torch.stack(image2_list).to(self.args.device)
                image2_features = self.clip_model.encode_image(image2)
    
                similarity = torch.cosine_similarity(image1_features, image2_features).cpu().tolist()

            return similarity
        else:
            raise NotImplementedError
        
    def get_similarity_from_features_batch(self, features1, features2_list):
        if self.method == 'clip' or self.method == 'dinov2':
            with torch.no_grad():
                # encode by clip
                features1 = features1.to(self.args.device)
                # # norm
                # features1 = features1 / features1.norm(dim=1, keepdim=True)

                # features2_list = [features2.to(self.args.device) for features2 in features2_list]
                # # norm
                # features2_list = [features2 / features2.norm(dim=1, keepdim=True) for features2 in features2_list]
                features2 = torch.cat(features2_list, dim=0).to(self.args.device)
    
                similarity = torch.cosine_similarity(features1, features2).cpu().tolist()
    
            return similarity
        else:
            raise NotImplementedError


def dict_to_csv(input_dict, out_path):
    # Extract the keys from the dictionary to use as column headers
    column_headers = list(input_dict[list(input_dict.keys())[0]].keys())
    sorted(column_headers)

    # Open the CSV file for writing
    with open(out_path, 'w', newline='') as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(['lvis'] + column_headers + ['avg'])

        avg_list = []

        # Write data rows
        for key, inner_dict in input_dict.items():
            value = [inner_dict[column] for column in column_headers]
            # get the average of list
            avg = sum(value) / len(value) if len(value) > 0 else 0
            avg_list.append(avg)
            row = [key] + value + [avg]
            writer.writerow(row)
        
        # write avg
        writer.writerow(['avg'] + [sum(avg_list) / len(avg_list) if len(avg_list) > 0 else 0])

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
    parser.add_argument('--lvis_crop_in_dir', type=str, )
    parser.add_argument('--dataset_json_path', type=str, )
    parser.add_argument('--gen_in_dir', type=str, )
    parser.add_argument('--result_out_dir', type=str, )
    parser.add_argument('--method', type=str, default='clip')
    parser.add_argument('--clip_ckpt_dir', type=str, default=None)

    parser.add_argument('--intra_category', action='store_true', default=False, help='whether to calculate similarity between the same category')

    parser.add_argument('--dist', action='store_true', default=False, help='whether to save cross attention maps')
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
    args.disable_init = True

    # print start datetime
    start_time = datetime.now()
    print('>>> Start datetime: {}'.format(str(start_time)))

    # init image similarity
    image_similarity = ImageSimilarity(method=args.method, args=args)

    with open(args.dataset_json_path, 'r') as f:
        gt_json = json.load(f)

    category_id_to_name = {}

    for category in gt_json['categories']:
        category_id = category['id']
        category_name = category['name']
        
        category_id_to_name[str(category_id - 1)] = category_name

    # list all category
    lvis_crop_in_dir = args.lvis_crop_in_dir
    gen_in_dir = args.gen_in_dir
    result_out_dir = args.result_out_dir

    lvis_in_categories = [dir_name for dir_name in os.listdir(lvis_crop_in_dir) if os.path.isdir(os.path.join(lvis_crop_in_dir, dir_name))]
    sorted(lvis_in_categories)
    # gen_in_categories = os.listdir(gen_in_dir)
    # sorted(gen_in_categories)

    if global_rank == 0  and os.path.exists(result_out_dir) == False:
        print('>>> Creating dir: {}'.format(result_out_dir))
        os.makedirs(result_out_dir)
    
    if args.dist:
        dist.barrier()

    for i, lvis_in_category in enumerate(lvis_in_categories):
        # split to each gpu
        if i % world_size != global_rank:
            continue

        # list all lvis images
        lvis_in_category_image_feature_paths = glob(os.path.join(lvis_crop_in_dir, lvis_in_category, '*.pt'))
        sorted(lvis_in_category_image_feature_paths)
        print('length of lvis_in_category_image_feature_paths: {}'.format(len(lvis_in_category_image_feature_paths)))

        # list all gen images
        if args.intra_category:
            gen_dir_name = lvis_in_category
        else:
            gen_dir_name = category_id_to_name[lvis_in_category]
        gen_in_category_image_feature_paths = glob(os.path.join(gen_in_dir, gen_dir_name, '*.pt'))
        sorted(gen_in_category_image_feature_paths)
        print('length of gen_in_category_image_feature_paths: {}'.format(len(gen_in_category_image_feature_paths)))

        if len(lvis_in_category_image_feature_paths) == 0:
            print('>>> No features in {}'.format(lvis_in_category))
            continue

        if len(gen_in_category_image_feature_paths) == 0:
            print('>>> No features in {}'.format(gen_dir_name))
            continue

        # create out dir
        current_result_out_dir = os.path.join(result_out_dir, lvis_in_category)

        if os.path.exists(current_result_out_dir) == False:
            print('>>> Creating dir: {}'.format(current_result_out_dir))
            os.mkdir(current_result_out_dir)

        gen_in_category_image_feature_list = []
        gen_in_category_basename_list = []

        # for each gen image
        for gen_in_category_image_feature_path in gen_in_category_image_feature_paths:
            # load image
            gen_in_category_image_feature = torch.load(gen_in_category_image_feature_path)
            gen_in_category_basename = os.path.basename(gen_in_category_image_feature_path).replace('.pt', '.png')

            gen_in_category_image_feature_list.append(gen_in_category_image_feature)
            gen_in_category_basename_list.append(gen_in_category_basename)

        # for each lvis image
        total_dict = {}
        for lvis_in_category_image_feature_path in lvis_in_category_image_feature_paths:
            print('>>> Processing {}'.format(lvis_in_category_image_feature_path))
            # load image
            lvis_in_category_image_feature = torch.load(lvis_in_category_image_feature_path)
            lvis_in_category_basename = os.path.basename(lvis_in_category_image_feature_path).replace('.pt', '.png')
            filename = os.path.splitext(lvis_in_category_basename)[0]

            total_result_out_path = os.path.join(current_result_out_dir, 'total.json')
            total_result_csv_out_path = os.path.join(current_result_out_dir, 'total.csv')
            if os.path.exists(total_result_csv_out_path):
                print('>>> Skip {}'.format(total_result_csv_out_path))
                continue

            # result
            out_dict = {}
            # calculate similarity
            similarity_list = image_similarity.get_similarity_from_features_batch(lvis_in_category_image_feature, gen_in_category_image_feature_list)

            for gen_in_category_basename, similarity in zip(gen_in_category_basename_list, similarity_list):
                out_dict[gen_in_category_basename] = similarity
        
            print(out_dict)
            total_dict[lvis_in_category_basename] = out_dict
            
            # current_result_out_path = os.path.join(current_result_out_dir, '{}.json'.format(filename))

            # print('>>> Saving result to {}'.format(current_result_out_path))
            # with open(current_result_out_path, 'w') as f:
            #     json.dump({lvis_in_category_basename: out_dict}, f)

        # save total result
        print('>>> Saving total result to {}'.format(total_result_out_path))
        with open(total_result_out_path, 'w') as f:
            json.dump(total_dict, f)

        # convert dict to csv
        print('>>> Saving total result to {}'.format(total_result_csv_out_path))
        
        dict_to_csv(total_dict, total_result_csv_out_path)
    
    if args.dist:
        dist.barrier()

    end_time = datetime.now()
    print('>>> End datetime: {}, used {}'.format(str(end_time), str(end_time - start_time)))

        
        