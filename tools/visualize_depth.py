import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
from tqdm import tqdm
# import torchviz
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map, visualize_depth
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
import sys

def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    parser.add_argument("--special", type=bool, default=False)
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()
    for data in tqdm(dataflow):
        metas = data["metas"].data[0][0]
        name = "{}-{}".format(metas["timestamp"], metas["token"])

        if args.mode == "pred":
            with torch.inference_mode():
                outputs = model(**data)
            if(False): # for displaying network structure
                model.train()
                outputs = model(return_loss=True, **data)
                # dot = torchviz.make_dot(outputs["loss/object/loss_heatmap"], params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
                dot = torchviz.make_dot(outputs["loss/object/loss_heatmap"], params=dict(model.named_parameters()))
                dot.format = 'svg'
                dot.render('viz/torchviz/structure_simple')
                sys.exit()

        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()
            if args.special:
                pred_depth = outputs[0]["pred_depth"].numpy()
                gt_depth = outputs[0]["gt_depth"].numpy()
                cam_points = outputs[0]["cam_points"]
            # print("depth_shapes", pred_depth.shape, gt_depth.shape, gt_depth.max(), gt_depth.min())

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        else:
            bboxes = None
            labels = None

        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(np.bool)
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
        else:
            masks = None
        index = 0
        # bboxes=None
        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):

                if args.special:
                    pred_depth_now = np.sum(pred_depth[index%6] * np.arange(1, 60, step=0.5).reshape(-1,1,1), axis=0)

                image = data["img"].data[0][0][k] #3,256,704

                if args.special:
                    visualize_depth(
                        os.path.join(args.out_dir, f"depth-{k}", f"{name}.png"),
                        image,
                        pred_depth_now,
                        gt_depth.squeeze()[index%6][...,None]
                    )
                index += 1
        some_points = []
        if args.special: # for camera point cloud viz (requires model to buffer the point cloud, search for visualize = False, change value to true in bevfusion.py and base.py)
            for k, single_cam_points in enumerate(cam_points):
                colors = ["white", "blue", "green", "yellow", "orange", "red"]
                visualize_lidar(
                    os.path.join(args.out_dir, f"cam_points_{k}", f"{name}.png"),
                    single_cam_points,
                    bboxes=bboxes,
                    labels=labels,
                    xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                    ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                    classes=cfg.object_classes,
                    point_color=colors[k]
                )
                some_points.append(single_cam_points[7])

        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
            )

        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )


if __name__ == "__main__":
    main()
