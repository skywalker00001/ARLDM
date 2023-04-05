# import sys
# import torch
# import os
# import torch.distributed as dist

# if __name__ == '__main__':
#     rank = int(sys.argv[1])
#     print(torch.cuda.nccl.version())
#     print(rank)
#     assert rank in [0,1]
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "29500"
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
#     dist.init_process_group(backend='nccl', rank=rank,world_size=2)
#     print("Finished init")

import torch.distributed as dist
import torch.utils.data.distributed
import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description='PyTorch distributed training on cifar-10')
parser.add_argument('--rank', default=1,type=int,
                    help='rank of current process')
parser.add_argument('--word_size', default=2,type=int,
                    help="word size")
parser.add_argument('--init_method', default='tcp://127.0.0.1:23456',
                    help="init-method")
args = parser.parse_args()

print('starting init_process_group')
dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank, world_size=args.word_size)
print('end init_process_group')