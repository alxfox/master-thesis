import os
from typing import Any, Dict, Tuple

import mmcv
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS
from PIL import Image
# import cv2

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations

from .loading_utils import load_augmented_point_cloud, reduce_LiDAR_beams
import torch
@PIPELINES.register_module()
class PointToMultiViewDepth(object):

    def __init__(self, dbound, downsample=1):
        self.downsample = downsample
        self.dbound = dbound

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.dbound[1]) & (
                    depth >= self.dbound[0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        ##
        imgs = results['img']
        #
        # imgs, rots, trans, intrins = results['img_inputs'][:4]
        # print(len(results["camera_intrinsics"]))
        # print(results["camera_intrinsics"][0].shape)
        # rots = results["camera2ego"][..., :3, :3]
        # trans = results["camera2ego"][..., :3, 3]
        intrins = torch.tensor(results["camera_intrinsics"][..., :3, :3])
        post_rots = torch.tensor(results["img_aug_matrix"][..., :3, :3])
        post_trans = torch.tensor(results["img_aug_matrix"][..., :3, 3])
        # post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]
            lidar2lidarego = torch.eye(4, dtype=torch.float32)
            # lidar2lidarego[:3, :3] = Quaternion(
            #     results['curr_img']['lidar2ego_rotation']).rotation_matrix
            # lidar2lidarego[:3, 3] = results['curr_img']['lidar2ego_translation']
            # lidar2lidarego = lidar2lidarego
            lidar2lidarego = torch.tensor(results["lidar2ego"])

            # lidarego2global = torch.eye(4, dtype=torch.float32)
            # lidarego2global[:3, :3] = Quaternion(
            #     results['curr_img']['ego2global_rotation']).rotation_matrix
            # lidarego2global[:3, 3] = results['curr_img']['ego2global_translation']
            # lidarego2global = lidarego2global
            lidarego2global = torch.tensor(results["ego2global"])

            # cam2camego = torch.eye(4, dtype=torch.float32)
            # cam2camego[:3, :3] = Quaternion(
            #     results['curr_img']['cams'][cam_name]
            #     ['sensor2ego_rotation']).rotation_matrix
            # cam2camego[:3, 3] = results['curr_img']['cams'][cam_name][
            #     'sensor2ego_translation']
            # cam2camego = cam2camego
            cam2camego = torch.tensor(results["camera2ego"][cid])
            # camego2global = torch.eye(4, dtype=torch.float32)
            # camego2global[:3, :3] = Quaternion(
            #     results['curr_img']['cams'][cam_name]
            #     ['ego2global_rotation']).rotation_matrix
            # camego2global[:3, 3] = results['curr_img']['cams'][cam_name][
            #     'ego2global_translation']
            # camego2global = camego2global
            camego2global = torch.tensor(results["ego2global"])

            # print("sheeep", [x.shape for x in [camego2global, lidar2lidarego, lidarego2global, cam2camego]])
            cam2img = torch.eye(4, dtype=torch.float32)
            cam2img = cam2img
            cam2img[:3, :3] = intrins[cid]

            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, results["img_shape"][1],
                                             results["img_shape"][0])
            # print("img shape for depth is", results["img_shape"])
            depth_map_list.append(depth_map)
            # print(len(imgs), results["img_shape"], depth_map.shape)
        depth_map = depth_map_list
        results["gt_depth"] = depth_map
        return results
@PIPELINES.register_module()
class LoadGTDepth:
    """Load depth by projecting LiDAR points into the camera frames

    Expects results['img'] to be loaded.
    Expects results['points'] to be loaded.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to generate ground truth depths for each camera frame

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # img is of shape (h, w, c, num_views)
        # modified for waymo
        images = []
        h, w = 0, 0
        for name in filename:
            images.append(Image.open(name))
        
        lidar2point = data["lidar_aug_matrix"]
        point2lidar = np.linalg.inv(lidar2point)
        lidar2ego = data["lidar2ego"]
        ego2global = data["ego2global"]
        lidar2global = ego2global @ lidar2ego @ point2lidar
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array

        
        results["gt_depth"] = depths
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles:
    """Load multi channel images from a list of separate channel files.

    Expects results['image_paths'] to be a list of filenames.
    When using 4d camera data, instead expects results['curr_img'] and results['adjacent_imgs'] to be filled.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged", camera_4d=False):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.camera_4d = camera_4d # when using 4d camera data, 

    def get_sensor_transforms(self, cam_info, cam_name, info, global2keylidar=None):
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']

        # sweep sensor to sweep ego is replaced with sensor to lidar
        # camera to lidar transform
        camera2lidar = np.eye(4).astype(np.float32)
        camera2lidar[:3, :3] = cam_info['cams'][cam_name]["sensor2lidar_rotation"]
        camera2lidar[:3, 3] = cam_info['cams'][cam_name]["sensor2lidar_translation"]

        # sensor2ego_rot = torch.Tensor(
        #     Quaternion(w, x, y, z).rotation_matrix)
        # sensor2ego_tran = torch.Tensor(
        #     cam_info['cams'][cam_name]['sensor2ego_translation'])
        # sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        # sensor2ego[3, 3] = 1
        # sensor2ego[:3, :3] = sensor2ego_rot
        # sensor2ego[:3, -1] = sensor2ego_tran



        # sweep ego to global
        # w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        # ego2global_rot = np.array(
        #     Quaternion(w, x, y, z).rotation_matrix)
        # ego2global_tran = np.array(
        #     cam_info['cams'][cam_name]['ego2global_translation'])
        # ego2global = ego2global_rot.new_zeros((4, 4))
        # ego2global[3, 3] = 1
        # ego2global[:3, :3] = ego2global_rot
        # ego2global[:3, -1] = ego2global_tran


        # return sensor2ego, ego2global
        #TODO probably need aug here
        # lidar2point = data["lidar_aug_matrix"]
        # point2lidar = np.linalg.inv(lidar2point)
        lidar2ego = info["lidar2ego"]
        ego2global = info["ego2global"]
        lidar2global = ego2global @ lidar2ego #@ point2lidar
        if global2keylidar is not None:
            camera2lidar = global2keylidar @ lidar2global @ camera2lidar # camera2lidar means camera to the lidar of curr_img

        return camera2lidar, lidar2global

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        # sensor2egos = []
        sensor2lidars = []
        lidar2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        # cam_names = self.choose_cams()
        # cam_names = results['cam_names']
        cam_names = []
        canvas = []
        
        for cam_name, cam_data in results['curr_img']['cams'].items():
            cam_names.append(cam_name)
            cam_data = cam_data
            filename = cam_data['data_path']
            img = Image.open(filename)
            post_rot = np.eye(2)
            post_tran = np.zeros(2)

            intrin = np.array(cam_data['camera_intrinsics'], dtype=np.float32)

            # sensor2ego, ego2global = \
            #     self.get_sensor_transforms(results['curr_img'], cam_name)
            sensor2lidar, lidar2global = \
                self.get_sensor_transforms(results['curr_img'], cam_name, results)
            global2keylidar = np.linalg.inv(lidar2global)
            # image view augmentation (resize, crop, horizontal flip, rotate)
            # img_augs = self.sample_augmentation(
            #     H=img.height, W=img.width, flip=flip, scale=scale)
            # resize, resize_dims, crop, flip, rotate = img_augs
            # img, post_rot2, post_tran2 = \
            #     self.img_transform(img, post_rot,
            #                        post_tran,
            #                        resize=resize,
            #                        resize_dims=resize_dims,
            #                        crop=crop,
            #                        flip=flip,
            #                        rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            # post_tran = np.zeros(3)
            # post_rot = np.eye(3)
            # post_tran[:2] = post_tran2
            # post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))
            imgs.append(img)

            # if True: #self.sequential:
            #     assert 'adjacent_imgs' in results
            #     for adj_info in results['adjacent_imgs']:
            #         filename_adj = adj_info['cams'][cam_name]['data_path']
            #         img_adjacent = Image.open(filename_adj)
            #         imgs.append(img_adjacent)
                    # TODO this one line was uncommented ^
            intrins.append(intrin)
            # sensor2egos.append(sensor2ego)
            sensor2lidars.append(sensor2lidar)
            lidar2globals.append(lidar2global)
            # post_rots.append(post_rot)
            # post_trans.append(post_tran)
        
        # print("post", cam_names)
        #TODO uncomment
        if True: #self.sequential:
            for adj_info in results['adjacent_imgs']:
                # post_trans.extend(post_trans[:len(cam_names)])
                # post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                for cam_name in cam_names:
                    # sensor2ego, ego2global = \
                    #     self.get_sensor_transforms(adj_info, cam_name)
                    sensor2lidar, lidar2global = \
                        self.get_sensor_transforms(adj_info, cam_name, results, global2keylidar=global2keylidar)
                    # sensor2egos.append(sensor2ego)
                    sensor2lidars.append(sensor2lidar)
                    lidar2globals.append(lidar2global)
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    imgs.append(img_adjacent)

        size = imgs[0].size
        # imgs = np.stack(imgs)

        # sensor2egos = torch.stack(sensor2egos)
        # sensor2lidars = np.stack(sensor2lidars)
        # lidar2globals = np.stack(lidar2globals)
        # intrins = np.stack(intrins)
        # post_rots = np.stack(post_rots)
        # post_trans = np.stack(post_trans)
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        # [1600, 900]
        results["img_shape"] = size
        results["ori_shape"] = size
        # Set initial values for default meta_keys
        results["pad_shape"] = size
        results["scale_factor"] = 1.0
        results['canvas'] = canvas
        intrins = np.stack(intrins, axis=0)
        results["camera_intrinsics"] = intrins
        # results["lidar2global"] = lidar2globals
        results["camera2lidar"] = sensor2lidars
        # results["img"] = imgs[:6] this didn't fix it
        results["img"] = imgs

        # return (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans)
        return # (imgs, sensor2lidars, ego2globals, intrins)

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        if self.camera_4d:
            self.get_inputs(results)
            results["filename"] = results["image_paths"]
        else:
            filename = results["image_paths"]
            # img is of shape (h, w, c, num_views)
            # modified for waymo
            images = []
            h, w = 0, 0
            for name in filename:
                images.append(Image.open(name))
            
            #TODO: consider image padding in waymo

            results["filename"] = filename
            # unravel to list, see `DefaultFormatBundle` in formating.py
            # which will transpose each image separately and then stack into array
            results["img"] = images
            # [1600, 900]
            results["img_shape"] = images[0].size
            results["ori_shape"] = images[0].size
            # Set initial values for default meta_keys
            results["pad_shape"] = images[0].size
            results["scale_factor"] = 1.0
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps:
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmcv.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results["points"]
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"] / 1e6
        if self.pad_empty_sweeps and len(results["sweeps"]) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results["sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["sweeps"]))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                # NOTE: seems possible to load frame -11?
                if not self.load_augmented:
                    choices = np.random.choice(
                        len(results["sweeps"]), self.sweeps_num, replace=False
                    )
                else:
                    # don't allow to sample the earliest frame, match with Tianwei's implementation.
                    choices = np.random.choice(
                        len(results["sweeps"]) - 1, self.sweeps_num, replace=False
                    )
            for idx in choices:
                sweep = results["sweeps"][idx]
                points_sweep = self._load_points(sweep["data_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                # TODO: make it more general
                if self.reduce_beams and self.reduce_beams < 32:
                    points_sweep = reduce_LiDAR_beams(points_sweep, self.reduce_beams)

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep["timestamp"] / 1e6
                points_sweep[:, :3] = (
                    points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
                )
                points_sweep[:, :3] += sweep["sensor2lidar_translation"]
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results["points"] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"


@PIPELINES.register_module()
class LoadBEVSegmentation:
    def __init__(
        self,
        dataset_root: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        classes: Tuple[str, ...],
    ) -> None:
        super().__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = classes

        self.maps = {}
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(dataset_root, location)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        lidar2point = data["lidar_aug_matrix"]
        point2lidar = np.linalg.inv(lidar2point)
        lidar2ego = data["lidar2ego"]
        ego2global = data["ego2global"]
        lidar2global = ego2global @ lidar2ego @ point2lidar

        map_pose = lidar2global[:2, 3]
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])

        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        patch_angle = yaw / np.pi * 180 - 90# + 180

        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        location = data["location"]
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        # masks = masks[:, ::-1, :].copy()
        masks = masks.transpose(0, 2, 1)
        masks = masks.astype(np.bool)

        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.long)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1
        # print("shape of bevx",labels.shape)
        data["gt_masks_bev"] = labels
        return data

@PIPELINES.register_module()
class LoadMap:
    def __init__(
        self,
        dataset_root: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        classes: Tuple[str, ...],
    ) -> None:
        super().__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(ybound[2])
        canvas_w = int(xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = classes

        self.maps = {}
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(dataset_root, location)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        lidar2point = data["lidar_aug_matrix"]
        point2lidar = np.linalg.inv(lidar2point)
        lidar2ego = data["lidar2ego"]
        ego2global = data["ego2global"]
        lidar2global = ego2global @ lidar2ego @ point2lidar
        map_pose = lidar2global[:2, 3]
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        # print("map_pose is", map_pose)
        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        patch_angle = yaw / np.pi * 180 - 90# + 180

        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        location = data["location"]
        # print("loc", location)
        # TODO
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        # masks = masks[:, ::-1, :].copy()
        masks = masks.transpose(0, 2, 1)
        masks = masks.astype(np.bool)

        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.float32)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1
        data["feature_map"] = labels
        # map_classes = [
        #     "drivable_area",
        #     # - drivable_area*
        #     "ped_crossing",
        #     "walkway",
        #     "stop_line",
        #     "carpark_area",
        #     # - road_divider
        #     # - lane_divider
        #     "divider"]
        # MAP_PALETTE = {
        #     "drivable_area": (166, 206, 227),
        #     "road_segment": (31, 120, 180),
        #     "road_block": (178, 223, 138),
        #     "lane": (51, 160, 44),
        #     "ped_crossing": (251, 154, 153),
        #     "walkway": (227, 26, 28),
        #     "stop_line": (253, 191, 111),
        #     "carpark_area": (255, 127, 0),
        #     "road_divider": (202, 178, 214),
        #     "lane_divider": (106, 61, 154),
        #     "divider": (106, 61, 154),
        # }
        # canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
        # canvas[:] = labels

        # for k, name in enumerate(map_classes):
        #     if name in MAP_PALETTE:
        #         canvas[masks[k], :] = MAP_PALETTE[name]
        # canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        # fpath = os.path.join("test_vis", "map", "abc.png")#f"{abc}.png")
        # mmcv.mkdir_or_exist(os.path.dirname(fpath))
        # mmcv.imwrite(canvas, fpath)
        # print("shape of map", labels.shape)
        # print("identical in loading:", np.allclose(data["gt_masks_bev"], labels))
        return data

@PIPELINES.register_module()
class LoadPointsFromFile:
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmcv.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        lidar_path = results["lidar_path"]
        points = self._load_points(lidar_path)
        points = points.reshape(-1, self.load_dim)
        # TODO: make it more general
        if self.reduce_beams and self.reduce_beams < 32:
            points = reduce_LiDAR_beams(points, self.reduce_beams)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )
        results["points"] = points

        return results


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
    """

    def __init__(
        self,
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_bbox=False,
        with_label=False,
        with_mask=False,
        with_seg=False,
        with_bbox_depth=False,
        poly2mask=True,
    ):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
        )
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results["gt_bboxes_3d"] = results["ann_info"]["gt_bboxes_3d"]
        results["bbox3d_fields"].append("gt_bboxes_3d")
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results["centers2d"] = results["ann_info"]["centers2d"]
        results["depths"] = results["ann_info"]["depths"]
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results["gt_labels_3d"] = results["ann_info"]["gt_labels_3d"]
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results["attr_labels"] = results["ann_info"]["attr_labels"]
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)

        return results
