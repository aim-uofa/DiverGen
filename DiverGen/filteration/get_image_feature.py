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
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from datetime import timedelta, datetime
from glob import glob
import csv

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class ImageSimilarity():
    def __init__(self, method = 'clip', args = None):
        self.method = method
        self.args = args
        if self.method == 'clip':
            # load clip
            print('>>> Loading CLIP...')
            if args.ckpt_dir is None:
                self.clip_model, self.preprocess = clip.load("ViT-L/14", device=args.device)
            else:
                self.clip_model, self.preprocess = clip.load("ViT-L/14", device=args.device, download_root=args.ckpt_dir)
            self.clip_model.float()
        elif self.method == 'dinov2':
            # load dinov2
            print('>>> Loading DINOv2...')
            if args.ckpt_dir is None:
                self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(args.device)
            else:
                self.dinov2 = torch.hub.load(args.ckpt_dir, 'dinov2_vitg14', source='local', pretrained=False)
                self.dinov2.load_state_dict(torch.load(os.path.join(args.ckpt_dir, 'dinov2_vitg14_pretrain.pth')))
                self.dinov2.to(args.device)
                
            
            # preprocess for dino v2
            self.preprocess = Compose([
                        Resize(224, interpolation=BICUBIC),
                        CenterCrop(224),
                        ToTensor(),
                        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ])
            self.dinov2.float()
            self.dinov2.eval()
       
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
        
    def get_feature_batch(self, image_list):
        if self.method == 'clip':
            with torch.no_grad():
                # encode by clip
                image_list = [self.preprocess(image) for image in image_list]
                image = torch.stack(image_list).to(self.args.device)
                image_features = self.clip_model.encode_image(image)
    
            return image_features
        else:
            raise NotImplementedError
    
    def get_feature(self, image):
        if self.method == 'clip':
            with torch.no_grad():
                # encode by clip
                image = self.preprocess(image).unsqueeze(0).to(self.args.device)
                image_features = self.clip_model.encode_image(image)
    
                return image_features
        elif self.method == 'dinov2':
            with torch.no_grad():
                # encode by dinov2
                image = self.preprocess(image).unsqueeze(0).to(self.args.device)
                image_features = self.dinov2(image)
    
                return image_features
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
    parser.add_argument('--process_lvis', action='store_true', default=False, help='whether to process lvis images')
    parser.add_argument('--lvis_crop_in_dir', type=str, )
    parser.add_argument('--dataset_json_path', type=str, )
    parser.add_argument('--process_gen', action='store_true', default=False, help='whether to process gen images')
    parser.add_argument('--gen_in_dir', type=str, )
    parser.add_argument('--gen_mask_in_dir', type=str, )
    parser.add_argument('--result_out_dir', type=str, )
    parser.add_argument('--method', type=str, default='dinov2')
    parser.add_argument('--ckpt_dir', type=str, default=None)

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

    if args.process_lvis:
        for i, lvis_in_category in enumerate(lvis_in_categories):
            # split to each gpu
            if i % world_size != global_rank:
                continue
            
            # list all lvis images
            lvis_in_category_image_paths = glob(os.path.join(lvis_crop_in_dir, lvis_in_category, '*.png'))
            sorted(lvis_in_category_image_paths)
            print('length of lvis_in_category_image_paths: {}'.format(len(lvis_in_category_image_paths)))
    
            if len(lvis_in_category_image_paths) == 0:
                print('>>> No images in {}'.format(lvis_in_category))
                continue
    
            # create out dir
            current_lvis_out_dir = os.path.join(result_out_dir, 'lvis', lvis_in_category)
    
            if os.path.exists(current_lvis_out_dir) == False:
                print('>>> Creating dir: {}'.format(current_lvis_out_dir))
                os.makedirs(current_lvis_out_dir)
    
            lvis_in_category_image_list = []
    
            # for each lvis image
            for lvis_in_category_image_path in lvis_in_category_image_paths:
                print('>>> Processing {}'.format(lvis_in_category_image_path))
                # load image
                lvis_in_category_image = Image.open(lvis_in_category_image_path).convert('RGB')
                lvis_in_category_basename = os.path.basename(lvis_in_category_image_path)
                filename = os.path.splitext(lvis_in_category_basename)[0]
    
                image_features = image_similarity.get_feature(lvis_in_category_image).cpu()
    
                # save feature
                current_lvis_out_path = os.path.join(current_lvis_out_dir, '{}.pt'.format(filename))
                print('>>> Saving result to {}'.format(current_lvis_out_path))
                torch.save(image_features, current_lvis_out_path)
    
    if args.process_gen:
        for i, lvis_in_category in enumerate(lvis_in_categories):
            # split to each gpu
            if i % world_size != global_rank:
                continue
    
            # list all gen images
            gen_dir_name = category_id_to_name[lvis_in_category]
            gen_in_category_image_paths = glob(os.path.join(gen_in_dir, gen_dir_name, '*.png'))
            sorted(gen_in_category_image_paths)
            print('length of gen_in_category_image_paths: {}'.format(len(gen_in_category_image_paths)))
    
            if len(gen_in_category_image_paths) == 0:
                print('>>> No images in {}'.format(gen_dir_name))
                continue
            
            current_gen_out_dir = os.path.join(result_out_dir, 'gen', gen_dir_name)
            
            if os.path.exists(current_gen_out_dir) == False:
                print('>>> Creating dir: {}'.format(current_gen_out_dir))
                os.makedirs(current_gen_out_dir)
            
            lvis_in_category_basename_list = []
            
            # for each gen image
            for gen_in_category_image_path in gen_in_category_image_paths:
                print('>>> Processing {}'.format(gen_in_category_image_path))
                gen_mask_in_category_image_path = os.path.join(args.gen_mask_in_dir, gen_dir_name, os.path.basename(gen_in_category_image_path))
                # load image
                gen_in_category_image = Image.open(gen_in_category_image_path).convert('RGB')
                # load mask
                gen_mask = Image.open(gen_mask_in_category_image_path).convert('L')

                # remove the background
                image_np = np.array(gen_in_category_image)
                gen_mask_np = np.array(gen_mask)
                image_np[gen_mask_np == 0] = 0
                gen_in_category_image = Image.fromarray(image_np)

                gen_in_category_basename = os.path.basename(gen_in_category_image_path)
                filename = os.path.splitext(gen_in_category_basename)[0]
    
                image_features = image_similarity.get_feature(gen_in_category_image).cpu()
    
                # save feature
                current_gen_out_path = os.path.join(current_gen_out_dir, '{}.pt'.format(filename))
                print('>>> Saving result to {}'.format(current_gen_out_path))
                torch.save(image_features, current_gen_out_path)

    if args.dist:
        dist.barrier()

    end_time = datetime.now()
    print('>>> End datetime: {}, used {}'.format(str(end_time), str(end_time - start_time)))

        
        