import os
import argparse
import json
from datetime import timedelta
from glob import glob

from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
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
        '--prompt',
        type=str,
        nargs='?',
        default='a painting of a virus monster playing guitar',
        help='the prompt to render'
    )
    parser.add_argument(
        '--from_file',
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
        '--n_samples',
        type=int,
        default=1,
        help='how many samples to produce for each given prompt. A.k.a. batch size',
    )
    parser.add_argument(
        '--max_batch_size',
        type=int,
        default=1,
        help='how many images to process at a time. Reduce this if running out of memory',
    )
    # parser.add_argument(
    #     '--stage_3_mini_batch_size',
    #     type=int,
    #     default=1,
    #     help='how many images to process at a time in stage 3. Reduce this if running out of memory',
    # )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='the seed (for reproducible sampling)',
    )
    parser.add_argument(
        '--dist',
        action='store_true',
        default=False,
        help='whether to save cross attention maps',
    )
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default='models/ldm/stable-diffusion-v1/',
        help='path to checkpoint of model',
    )
    parser.add_argument(
        '--dataset_json_path', 
        type=str, 
        default='/data/datasets/lvis/lvis_v1_val.json'
    )
    parser.add_argument(
        '--stages',
        type=str,
        nargs='+',
        default="'I' 'II'",
        help='model stages to use',
    )
    parser.add_argument(
        '--offset',
        type=int,
        default=1024,
        help='offset for the prompt',
    )
    parser.add_argument(
        '--disable_overwrite',
        action='store_true',
        help='disable overwrite',
        default=False
    )

    args = parser.parse_args()

    # # init distributed
    if args.dist:
        global_rank, local_rank, world_size, device = init_distributed()
    else:
        global_rank = 0
        local_rank = 0
        world_size = 1
        device = 'cuda:0'
    print('local rank: {}, global rank: {}, world size: {}, device: {}'.format(local_rank, global_rank, world_size, device))

    torch.cuda.set_device(device)

    total_batch_size = args.n_samples // world_size
    assert total_batch_size * world_size == args.n_samples, 'n_samples must be divisible by world_size'

    batch_size = total_batch_size // args.max_batch_size
    remainder_batch_size = total_batch_size % args.max_batch_size
    assert batch_size * args.max_batch_size + remainder_batch_size == total_batch_size, 'batch size error, batch_size: {}, args.max_batch_size: {}, remainder_batch_size: {}, total_batch_size: {}'.format(batch_size, args.max_batch_size, remainder_batch_size, total_batch_size)
    if remainder_batch_size > 0:
        batch_size += 1

    outpath = args.outdir
    sample_path = os.path.join(outpath, 'samples')

    if 'I' in args.stages:
        # stage 1
        print('==> Loading stage I from {}...'.format(os.path.join(args.ckpt_dir, 'IF-I-XL-v1.0')))
        stage_1 = DiffusionPipeline.from_pretrained(os.path.join(args.ckpt_dir, 'IF-I-XL-v1.0'), variant='fp16', torch_dtype=torch.float16)
        # stage_1 = DiffusionPipeline.from_pretrained('DeepFloyd/IF-I-XL-v1.0', variant='fp16', torch_dtype=torch.float16)
        stage_1.to(device)
        # stage_1.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
        stage_1.enable_model_cpu_offload(local_rank)

        # create dir
        if global_rank == 0:
            current_dir = os.path.join(sample_path, 'I')
    
            if os.path.exists(current_dir) == False:
                os.makedirs(current_dir)
                        
        if args.dist:
            dist.barrier()

    if 'II' in args.stages:
        # stage 2
        print('==> Loading stage II from {}...'.format(os.path.join(args.ckpt_dir, 'IF-II-L-v1.0')))
        stage_2 = DiffusionPipeline.from_pretrained(
            os.path.join(args.ckpt_dir, 'IF-II-L-v1.0'), text_encoder=None, variant='fp16', torch_dtype=torch.float16
        )
        # stage_2 = DiffusionPipeline.from_pretrained(
        #     'DeepFloyd/IF-II-L-v1.0', text_encoder=None, variant='fp16', torch_dtype=torch.float16
        # )
        stage_2.to(device)
        # stage_2.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
        stage_2.enable_model_cpu_offload(local_rank)

        if global_rank == 0:
            current_dir = os.path.join(sample_path, 'II')
    
            # create dir
            if os.path.exists(current_dir) == False:
                os.makedirs(current_dir)
        
        if args.dist:
            dist.barrier()
    
    if 'III' in args.stages:
        # stage 3
        print('==> Loading stage III...')
        safety_modules = {'feature_extractor': stage_1.feature_extractor, 'safety_checker': stage_1.safety_checker, 'watermarker': None}
        stage_3 = DiffusionPipeline.from_pretrained(
            os.path.join(args.ckpt_dir, 'stable-diffusion-x4-upscaler'), **safety_modules, 
            torch_dtype=torch.float16)
        stage_3.to(device)
        stage_3.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
        stage_3.enable_model_cpu_offload(local_rank)


        if global_rank == 0:
            # create dir
            current_dir = os.path.join(sample_path, 'III')
    
            if os.path.exists(current_dir) == False:
                os.makedirs(current_dir)
        
        if args.dist:
            dist.barrier()
    
    generator = torch.manual_seed(args.seed + global_rank)

    if not args.from_file:
        prompt = args.prompt
        assert prompt is not None
        data = batch_size * [prompt]
    else:
        if os.path.isdir(args.from_file[0]):
            args.from_file = glob(os.path.join(args.from_file[0], '*.txt'))
            sorted(args.from_file)
        
        total_length = len(args.from_file)

        for current_i, prompt_file_path in enumerate(args.from_file):
            data = []
            print('==> Reading prompts from {}, {}/{}'.format(prompt_file_path, current_i + 1, total_length))
            with open(prompt_file_path, 'r') as f:
                data.extend(f.read().splitlines())
            
            category_id = os.path.basename(prompt_file_path).split('.')[0]                

            data = batch_size * data
    
            # sort to make the same prompts near
            data = sorted(data)

            for i, prompt in enumerate(data):
                prompt = prompt.strip()
                print('==> Generating {} Prompt: {}'.format(i, prompt))
                
                # first batch of category
                if i % batch_size == 0:
                    tmp = 0
                    if remainder_batch_size != 0:
                        current_num_images_per_prompt = remainder_batch_size
                    else:
                        current_num_images_per_prompt = args.max_batch_size
                else:
                    current_num_images_per_prompt = args.max_batch_size
        
                if 'I' in args.stages:
        
                    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)
        
                    # stage 1
                    if args.disable_overwrite:
                        count = (current_num_images_per_prompt - 1) + i * current_num_images_per_prompt + total_batch_size * global_rank + args.offset + (i // batch_size) * args.n_samples
                        filename = '{}_{:07d}.png'.format(category_id, count)
                        out_path = os.path.join(sample_path, 'I', filename)
        
                        if os.path.exists(out_path):
                            print('==> Skipping stage I for {}...'.format(filename))
                            continue
        
                    print('==> Running stage I for {}_{}...'.format(category_id, i))
                    image = stage_1(prompt_embeds=prompt_embeds, 
                                    negative_prompt_embeds=negative_embeds, 
                                    generator=generator, 
                                    output_type='pt',
                                    num_images_per_prompt=current_num_images_per_prompt).images
        
                    for j in range(current_num_images_per_prompt):
                        count = j + tmp + total_batch_size * global_rank + args.offset + (i // batch_size) * args.n_samples
                        filename = '{}_{:07d}.png'.format(category_id, count)
                        print('==> Saving stage I for {}...'.format(filename))
        
                        out_path = os.path.join(sample_path, 'I', filename)
                        pt_to_pil(image)[j].save(out_path)
        
                
                # repeat prompts
                prompt_embeds = prompt_embeds.repeat((current_num_images_per_prompt, 1, 1))
                negative_embeds = negative_embeds.repeat((current_num_images_per_prompt, 1, 1))
                
                if 'II' in args.stages:
                    # stage 2
                    if args.disable_overwrite:
                        count = (current_num_images_per_prompt - 1) + i * current_num_images_per_prompt + total_batch_size * global_rank + args.offset + (i // batch_size) * args.n_samples
                        filename = '{}_{:07d}.png'.format(category_id, count)
                        out_path = os.path.join(sample_path, 'II', filename)
        
                        if os.path.exists(out_path):
                            print('==> Skipping stage II for {}...'.format(filename))
                            continue
        
                    print('==> Running stage II for {}_{}...'.format(category_id, i))
                    image = stage_2(
                        image=image, 
                        prompt_embeds=prompt_embeds, 
                        negative_prompt_embeds=negative_embeds, 
                        generator=generator, 
                        output_type='pt',
                    ).images
        
                    for j in range(current_num_images_per_prompt):
                        count = j + tmp + total_batch_size * global_rank + args.offset + (i // batch_size) * args.n_samples
                        filename = '{}_{:07d}.png'.format(category_id, count)
                        print('==> Saving stage II for {}...'.format(filename))
        
                        out_path = os.path.join(sample_path, 'II', filename)
                        pt_to_pil(image)[j].save(out_path)
        
                
                if 'III' in args.stages:
                    if args.disable_overwrite:
                        count = (current_num_images_per_prompt - 1) + i * current_num_images_per_prompt + total_batch_size * global_rank + args.offset + (i // batch_size) * args.n_samples
                        filename = '{}_{:07d}.png'.format(category_id, count)
                        out_path = os.path.join(sample_path, 'III', filename)
        
                        if os.path.exists(out_path):
                            print('==> Skipping stage III for {}...'.format(filename))
                            continue
        
                    for j in range(current_num_images_per_prompt):
                        count = j + tmp + total_batch_size * global_rank + args.offset + (i // batch_size) * args.n_samples
                        filename = '{}_{:07d}.png'.format(category_id, count)
                        print('==> Saving stage III for {}...'.format(filename))
                    
                        # stage 3
                        print('==> Running stage III for {}...'.format(filename))
                        current_image = stage_3(prompt=prompt, 
                                        image=image[j:j+1], 
                                        generator=generator, 
                                        noise_level=100).images
                        out_path = os.path.join(sample_path, 'III', filename)
                        current_image[0].save(out_path)
        
                tmp += current_num_images_per_prompt

    print('done')