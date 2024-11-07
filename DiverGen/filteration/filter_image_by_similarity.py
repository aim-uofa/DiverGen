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

def filename_dict_to_csv(input_dict, out_path):
    # Extract the keys from the dictionary to use as column headers
    column_headers = list(input_dict[list(input_dict.keys())[0]].keys())
    sorted(column_headers)

    # Open the CSV file for writing
    with open(out_path, 'w', newline='') as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(['gen'] + column_headers + ['avg'])

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

def out_dict_to_csv(input_dict, out_path):
    # Extract the keys from the dictionary to use as column headers
    column_headers = ['category', 'filename', 'similarity']

    # Open the CSV file for writing
    with open(out_path, 'w', newline='') as csvfile:
        print('>>> Write to csv: {}'.format(out_path))
        # Create a CSV writer
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(column_headers)

        # Write data rows
        for key, value in input_dict.items():
            for filename, similarity in value.items():
                row = [key, filename, similarity]
                writer.writerow(row)
        
        csvfile.close()

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
    parser.add_argument('--result_in_dir', type=str, default='/data/datasets/lvis/val2017_padding_white_crop')
    parser.add_argument('--dataset_json_path', type=str, default='/data/datasets/lvis/lvis_v1_val.json')
    parser.add_argument('--result_out_dir', type=str, default='output/debug/230915_sim')
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--save_filtered_out', action='store_true', default=False)

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

    # print start datetime
    start_time = datetime.now()
    print('>>> Start datetime: {}'.format(str(start_time)))

    with open(args.dataset_json_path, 'r') as f:
        gt_json = json.load(f)

    category_id_to_name = {}

    for category in gt_json['categories']:
        category_id = category['id']
        category_name = category['name']
        
        category_id_to_name[str(category_id - 1)] = category_name

    # list all category
    result_in_dir = args.result_in_dir
    result_out_dir = args.result_out_dir
    
    result_out_path = os.path.join(result_out_dir, 'filename_thres_{}.csv'.format(args.threshold))

    if not os.path.exists(result_out_dir):
        print('>>> Create result out dir: {}'.format(result_out_dir))
        os.makedirs(result_out_dir)

    if os.path.exists(result_out_path):
        print('>>> Remove: {}'.format(result_out_path))
        os.remove(result_out_path)

    if args.save_filtered_out:
        out_filtered_out_dict = {}

    lvis_in_categories = [dir_name for dir_name in os.listdir(result_in_dir) if os.path.isdir(os.path.join(result_in_dir, dir_name))]
    sorted(lvis_in_categories)
    
    out_dict = {}

    for lvis_in_category in lvis_in_categories:
        filename_dict = {}
        print('>>> Processing category: {}'.format(lvis_in_category))
        json_in_path = os.path.join(result_in_dir, lvis_in_category, 'total.json')

        out_dir = os.path.join(result_out_dir, lvis_in_category)

        if not os.path.exists(out_dir):
            print('>>> Create result out dir: {}'.format(out_dir))
            os.makedirs(out_dir)

        json_out_path = os.path.join(out_dir, 'total_filename.json')
        csv_out_path = os.path.join(out_dir, 'total_filename.csv')

        if os.path.exists(json_out_path) \
            and os.path.exists(csv_out_path) \
            and args.overwrite == False:
            print('>>> Skip: {}'.format(json_out_path))
            continue

        with open(json_in_path, 'r') as f:
            json_dict = json.load(f)

        for lvis_filename, similarity_list in json_dict.items():
            for gen_filename, similarity in similarity_list.items():
                if gen_filename not in filename_dict:
                    filename_dict[gen_filename] = {}
                filename_dict[gen_filename][lvis_filename] = similarity

        # save to json
        if os.path.exists(json_out_path):
            print('>>> Remove: {}'.format(json_out_path))
            os.remove(json_out_path)
        
        with open(json_out_path, 'w') as f:
            json.dump(filename_dict, f)

        # save to csv
        if os.path.exists(csv_out_path):
            print('>>> Remove: {}'.format(csv_out_path))
            os.remove(csv_out_path)

        filename_dict_to_csv(filename_dict, csv_out_path)

    for lvis_in_category in lvis_in_categories:
        print('>>> Processing category: {}'.format(lvis_in_category))
        csv_in_path = os.path.join(result_out_dir, lvis_in_category, 'total_filename.csv')

        gen_category_name = category_id_to_name[lvis_in_category]

        if gen_category_name not in out_dict:
            out_dict[gen_category_name] = {}

        with open(csv_in_path, 'r') as f:
            reader = csv.reader(f)
            # get the first column as filename, and last column as similarity
            # skip header
            next(reader)
            for row in reader:
                # skip avg row
                if row[0] == 'avg':
                    continue

                filename = row[0]
                similarity = float(row[-1])

                if similarity >= args.threshold:
                    out_dict[gen_category_name][filename] = similarity
                else:
                    if args.save_filtered_out:
                        if gen_category_name not in out_filtered_out_dict:
                            out_filtered_out_dict[gen_category_name] = {}
                        out_filtered_out_dict[gen_category_name][filename] = similarity

    out_dict_to_csv(out_dict, result_out_path)

    if args.save_filtered_out:
        out_filtered_out_path = os.path.join(result_out_dir, 'filtered_out_filename_thres_{}.csv'.format(args.threshold))
        out_dict_to_csv(out_filtered_out_dict, out_filtered_out_path)

    end_time = datetime.now()
    print('>>> End datetime: {}, used {}'.format(str(end_time), str(end_time - start_time)))

        
        