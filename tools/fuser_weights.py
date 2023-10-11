import argparse
import copy
import os
import warnings

import mmcv
import torch
from torchpack.utils.config import configs
from torchpack import distributed as dist
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, save_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import recursive_eval
from nuscenes.utils.data_classes import RadarPointCloud

def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="old config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    args = parser.parse_args()
    return args
def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    # print(model)
    # print(model.fuser)
    fuser = model.fuser
    fuser_weights = super(type(fuser), fuser).__getitem__(0).weight
    print(fuser_weights.shape)
    spread = fuser_weights.abs().sum(0).sum(1).sum(1)
    spreadw = fuser_weights.sum(0).sum(1).sum(1)
    print(spread.shape)
    print(spread)
    print(spreadw)
    # print()
if __name__ == "__main__":
    main()
