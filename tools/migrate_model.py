import argparse
import copy
import os
import warnings

import mmcv
import torch
from torchpack.utils.config import Config as Conf
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
    parser.add_argument("new_config", help="new config filepath")
    parser.add_argument("out", help="output result file in pickle format")
    parser.add_argument("--lidar_config", help="lidar encoder config filepath")
    parser.add_argument("--lidar_checkpoint", help="lidar encoder checkpoint filepath")
    parser.add_argument("--camera_config", help="camera encoder config filepath")
    parser.add_argument("--camera_checkpoint", help="camera encoder checkpoint filepath")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where results will be saved")
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())
    # Create new base model from the config
    cfgs = Conf()
    cfgs.load(args.new_config, recursive=True)
    cfg_new = Config(recursive_eval(cfgs), filename=args.new_config)
    model_new = build_model(cfg_new.model)
    model_new.init_weights()

    if args.lidar_config != None and args.lidar_checkpoint != None:
        print("loading lidar encoder weights...")
        cfgs = Conf()
        cfgs.load(args.lidar_config, recursive=True)
        cfg_lidar = Config(recursive_eval(cfgs), filename=args.lidar_config)
    #     model_lidar = build_model(cfg_lidar.model)
    #     load_checkpoint(model_lidar, args.lidar_checkpoint, map_location="cpu")
    #     model_new.encoders.lidar.load_state_dict(model_lidar.encoders.lidar.state_dict())
    #     print("transferred lidar weights to new model")

    if args.camera_config != None and args.camera_checkpoint != None:
        print("loading camera encoder weights...")
        cfgs = Conf()
        cfgs.load(args.camera_config, recursive=True)
        cfg_camera = Config(recursive_eval(cfgs), filename=args.camera_config)
        model_camera = build_model(cfg_camera.model)
        load_checkpoint(model_camera, args.camera_checkpoint, map_location="cpu")
        model_new.encoders.camera.load_state_dict(model_camera.encoders.camera.state_dict())
        print("transferred camera weights to new model")
    # model_new.encoders.lidar.load_state_dict(model_old.encoders.lidar.state_dict())
    # model_new.decoder.load_state_dict(model_old.decoder.state_dict())
    # model_new.heads.load_state_dict(model_old.heads.state_dict())
    # model_new.fuser.load_state_dict(model_old.fuser.state_dict(),strict=False)
    # print(model_new.fuser.weight.data)
    # print(torch.allclose(model_old.fuser.weight.data, model_new.fuser.weight.data), torch.allclose(model_old.fuser.weight.data, model_old.fuser.weight.data))
    # print(model_new)
    # save_checkpoint(model_new, args.out)

if __name__ == "__main__":
    main()
