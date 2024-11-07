
import os
import json
import torch
import oss2
import torch.distributed as dist
from datetime import timedelta, datetime

from argparse import ArgumentParser

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
    parser.add_argument('--inst_pool_path', type=str, action='append', default=[])
    parser.add_argument('--enable_replace', action='store_true', default=False)
    parser.add_argument('--before_prefix', type=str, action='append', default=[])
    parser.add_argument('--after_prefix', type=str, action='append', default=[])
    parser.add_argument('--out_inst_pool_path', type=str, default='output/debug/230915_sim')

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

    # print start datetime
    start_time = datetime.now()
    print('>>> Start datetime: {}'.format(str(start_time)))

    length_of_inst_pool_path = len(args.inst_pool_path)

    assert length_of_inst_pool_path > 0, 'inst_pool_path is empty'
    if args.enable_replace:
        assert length_of_inst_pool_path == len(args.before_prefix), 'length of inst_pool_path and before_prefix should be equal'
        assert length_of_inst_pool_path == len(args.after_prefix), 'length of inst_pool_path and after_prefix should be equal'

    out_inst_pool = {}

    for i in range(length_of_inst_pool_path):
        current_inst_pool_path = args.inst_pool_path[i]
        if args.enable_replace:
            current_before_prefix = args.before_prefix[i]
            current_after_prefix = args.after_prefix[i]
        print('>>> Processing inst pool from {}'.format(current_inst_pool_path))
        with open(current_inst_pool_path, 'r') as f:
            current_inst_pool = json.load(f)

        for key, value in current_inst_pool.items():
            start_index = 0
            end_index = len(value)
            
            print('start index: {}, end index: {}'.format(start_index, end_index))

            if args.enable_replace:
                value = [v.replace(current_before_prefix, current_after_prefix) for v in value][start_index:end_index]

            if key in out_inst_pool:
                out_inst_pool[key].extend(value)
            else:
                out_inst_pool[key] = value
    

    dirname = os.path.dirname(args.out_inst_pool_path)
    if os.path.exists(dirname) == False:
        print('>>> Creating dir {}'.format(dirname))
        os.makedirs(dirname)

    print('>>> Saving inst pool to {}'.format(args.out_inst_pool_path))
    with open(args.out_inst_pool_path, 'w') as f:
        json.dump(out_inst_pool, f)

    end_time = datetime.now()
    print('>>> End datetime: {}, used {}'.format(str(end_time), str(end_time - start_time)))