import copy
import os
from typing import List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt

from ..bbox import LiDARInstance3DBoxes

__all__ = ["visualize_camera", "visualize_lidar", "visualize_map", "visualize_feature_map", "visualize_depth"]


OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}


def visualize_depth(
    fpath: str,
    image: np.ndarray,
    pred_depth: np.ndarray, # pred_depth is upscaled from 32x88x1 to 256x704x1
    gt_depth: np.ndarray, # gt_depth is 256x704x1
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
) -> None:
    # print(image.dtype, image.max(), image.min())
    image = np.array(image*255, dtype=np.uint8)#.astype(np.uint8)
    image = np.transpose(image, axes=(1,2,0))#.astype(np.uint8)
    max_depth = 59.5
    pred_canvas = np.clip((pred_depth*255/max_depth),a_min=0, a_max = 255).astype(np.uint8)
    pred_canvas = cv2.applyColorMap(pred_canvas, cv2.COLORMAP_JET) # blue = 0
    pred_canvas = cv2.resize(pred_canvas, image.shape[1::-1], cv2.INTER_AREA)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image = cv2.resize(image, (88, 32), cv2.INTER_AREA)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pred_canvas = np.concatenate((image, pred_canvas), axis=0)

    gt_canvas = np.clip((gt_depth*255/max_depth),a_min=0, a_max = 255).astype(np.uint8)
    # print(gt_depth.shape)
    gt_canvas = cv2.applyColorMap(gt_canvas, cv2.COLORMAP_JET) # blue = 0
    gt_canvas = cv2.resize(gt_canvas, image.shape[1::-1], cv2.INTER_AREA)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # # image = cv2.resize(image, (88, 32), cv2.INTER_AREA)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay_canvas = np.copy(gt_canvas)
    no_point_mask = np.logical_and(gt_canvas[...,0] == 128, gt_canvas[...,1] == 0)
    # print(gt_canvas.shape,no_point_mask.shape, gt_canvas[0, 100])
    overlay_canvas[no_point_mask] = image[no_point_mask]
    # print(overlay_canvas[no_point_mask][:,0])#image[no_point_mask]
    gt_canvas = np.concatenate((gt_canvas, overlay_canvas), axis=0)
    # canvas = image
    canvas = np.concatenate((pred_canvas,gt_canvas), axis=1)
    # canvas = np.concatenate((pred_canvas_, gt_canvas), axis=1)
    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)

def visualize_camera(
    fpath: str,
    image: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
) -> None:
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        labels = labels[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        labels = labels[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            for start, end in [
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 5),
                (3, 2),
                (3, 7),
                (4, 5),
                (4, 7),
                (2, 6),
                (5, 6),
                (6, 7),
            ]:
                cv2.line(
                    canvas,
                    coords[index, start].astype(np.int),
                    coords[index, end].astype(np.int),
                    color or OBJECT_PALETTE[name],
                    thickness,
                    cv2.LINE_AA,
                )
        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)


def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
    point_color="white"
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c=point_color,
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name]) / 255,
            )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def visualize_map(
    fpath: str,
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    assert masks.dtype == np.bool, masks.dtype

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)

def visualize_feature_map(
    fpath: str,
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    # assert masks.dtype == np.bool, masks.dtype
    for i in range(32):
        canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
        canvas[:] = (255,255,255)
        canvas = np.clip((canvas * np.expand_dims(masks[i], 2)), 0, 255).astype(np.uint8)
        # canvas[masks[i], :] = (0, 0, 0)
        print("cc", canvas.shape)
        print("mm", masks[i].mean(), masks[i].max())
        # canvas = np.clip(([[(255, 255, 255)]] * np.expand_dims(masks[i],2)),0,1).astype(np.uint)
        # print(canvas.shape)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

        mmcv.mkdir_or_exist(os.path.dirname(fpath))
        mmcv.imwrite(canvas, fpath + "_" + str(i)+".png")