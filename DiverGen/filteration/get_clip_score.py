import os
import argparse
import json
import clip
from datetime import timedelta
from glob import glob
from PIL import Image
from torch.cuda.amp import autocast as autocast

import torch
import torch.distributed as dist
import numpy as np
from torchvision import transforms

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False

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
    parser.add_argument('--indir', type=str)
    parser.add_argument('--outdir', type=str, nargs='?')
    parser.add_argument('--use_mask', action='store_true', default=False)
    parser.add_argument('--in_mask_dir', type=str)
    parser.add_argument('--seg_name', type=str)
    parser.add_argument('--dist', action='store_true', default=False)
    parser.add_argument('--n_samples', nargs='+', type=int)
    parser.add_argument('--max_batch_size', type=int, default=1)
    parser.add_argument('--in_lvis_json_path', type=str, default='output/220714_lvis_v1_t5_with_original_prompts_id/prompts/lvis_v1_id_to_prompt.json')
    parser.add_argument('--clip_ckpt_dir', type=str, default=None)
    parser.add_argument('--stages', type=str, nargs='+')

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

    # load clip
    print('>>> Loading CLIP...')
    if args.clip_ckpt_dir is None:
        clip_model, preprocess = clip.load("ViT-L/14", device=device)
    else:
        clip_model, preprocess = clip.load("ViT-L/14", device=device, download_root=args.clip_ckpt_dir)
    clip_model.float()

    if args.use_mask:
        n_px = 224
        preprocess = transforms.Compose([
            transforms.Resize(n_px, interpolation=BICUBIC),
            transforms.CenterCrop(n_px),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    with open(args.in_lvis_json_path, 'r') as f:
        data = json.load(f)

    # check the number of samples
    for stage in args.stages:
        if stage == 'sd':
            current_in_dir = args.indir
            if args.use_mask:
                current_out_dir = os.path.join(args.outdir, args.seg_name)
            else:
                current_out_dir = args.outdir
        else:
            current_in_dir = os.path.join(args.indir, stage)
            if args.use_mask:
                current_out_dir = os.path.join(args.outdir, stage, args.seg_name)
            else:
                current_out_dir = os.path.join(args.outdir, stage)

        for category in data:
            category_name = category['name']
            sample_dir = os.path.join(current_in_dir, category_name)
            sample_paths = sorted(glob(os.path.join(sample_dir, '*.png')))
            len_samples = len(sample_paths)
            if len_samples not in args.n_samples:
                print('>>>Skip {}, it has {} images, but expected to have {}'.format(category_name, len_samples, args.n_samples))
                category['clip_scores'] = []
                if args.use_mask:
                    category['areas'] = []
                continue

            
            picked_image_index_and_paths = []

            for i, stage_sample_path in enumerate(sample_paths):
                if i % world_size == global_rank:
                    picked_image_index_and_paths.append((i, stage_sample_path))

            clips = []
            indices = []
            batch_images = []

            if args.use_mask:
                areas = []
            
            print('>>> Processing {}...'.format(category_name))
            with autocast(enabled=False):
                for (i, stage_sample_path) in picked_image_index_and_paths:
                    # read image
                    image = Image.open(stage_sample_path).convert("RGB")

                    if args.use_mask:
                        if stage == 'sd':
                            mask_path = os.path.join(args.in_mask_dir, args.seg_name, category_name, os.path.basename(stage_sample_path))
                        else:
                            mask_path = os.path.join(args.in_mask_dir, stage, args.seg_name, category_name, os.path.basename(stage_sample_path))
                        mask = Image.open(mask_path).convert("L")
                        mask = np.expand_dims(np.array(mask), axis=2)
                        mask_im = mask > 128
                        
                        image = image * mask_im + np.ones_like(image) * (1 - mask_im)
                        image = Image.fromarray(image.astype(np.uint8))

                        mask_area = np.sum(mask_im) / mask_im.shape[0] / mask_im.shape[1]
                        mask_areas = mask_area.tolist()
                        areas.append(mask_area)

                    batch_images.append(preprocess(image))
                    indices.append(i)
                    if len(batch_images) < args.max_batch_size and i != picked_image_index_and_paths[-1][0]:
                        continue
    
                    # get clip score
                    text='a photo of a single {}'.format(' '.join(category_name.split('_')))
                    text_feature = clip.tokenize(text).to(device)

                    _, logits_per_text = clip_model(torch.stack(batch_images, dim=0).to(device), text_feature)
                    logits_per_text = logits_per_text.view(-1).cpu().tolist()
    
                    # save clip score
                    for j, clip_score in enumerate(logits_per_text):
                        clips.append(clip_score)
    
                    # reset batch images
                    batch_images.clear()
    
    
                if world_size > 1:
                    # convert list to tensor
                    indices_tensor = torch.tensor(indices).to(device)
                    clips_tensor = torch.tensor(clips, dtype=torch.float32).to(device)

                    if args.use_mask:
                        areas_tensor = torch.tensor(areas, dtype=torch.float32).to(device)
                            
                    # gather tensors from all gpu
                    gathered_indices_tensor = [torch.zeros_like(indices_tensor) for _ in range(world_size)]
                    gathered_clips_tensor = [torch.zeros_like(clips_tensor, dtype=torch.float32) for _ in range(world_size)]
                    if args.use_mask:
                        gathered_areas_tensor = [torch.zeros_like(areas_tensor, dtype=torch.float32) for _ in range(world_size)]

                    dist.all_gather(gathered_indices_tensor, indices_tensor)
                    dist.all_gather(gathered_clips_tensor, clips_tensor)
                    if args.use_mask:
                        dist.all_gather(gathered_areas_tensor, areas_tensor)
        
                    # cat all tensors
                    gathered_indices_tensor_flat = torch.cat(gathered_indices_tensor)
                    gathered_clips_tensor_flat = torch.cat(gathered_clips_tensor)
                    if args.use_mask:
                        gathered_areas_tensor_flat = torch.cat(gathered_areas_tensor)
                    
                    # sort tensors
                    _, sorted_indices = gathered_indices_tensor_flat.sort()
                    clips = gathered_clips_tensor_flat[sorted_indices]
                    if args.use_mask:
                        areas = gathered_areas_tensor_flat[sorted_indices]
                    
                    # convert tensor to list
                    clips = clips.tolist()
                    if args.use_mask:
                        areas = areas.tolist()
    
                category['clip_scores'] = clips
                if args.use_mask:
                    category['areas']=areas
        
        if global_rank == 0:
            # save category
            with open(os.path.join(current_out_dir, "results.json"), 'w') as f:
                json.dump(data, f)
        
    print('done')