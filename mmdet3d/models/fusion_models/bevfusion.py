from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F
# from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
# import numpy as np
# import mmcv
from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS


from .base import Base3DFusionModel

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        freeze_state={ "lidar": False, "camera": False },
        use_aux_loss=[],
        merge_bev_4d_cfg: Dict[str, Any] = {"type": "first"},
        loss_depth_weight=0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.loss_depth_weight = loss_depth_weight
        self.merge_bev_4d_cfg = merge_bev_4d_cfg
        self.use_aux_loss = use_aux_loss
        # if("camera" in use_aux_loss):
        #     self.aux_loss_encoder = build_fuser({"type": "ConvFuser", "in_channels": [80], "out_channels": 256})
        self.freeze_state = freeze_state
        self.encoders = nn.ModuleDict()
        self.visualize = False
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)
        # TODO
        if encoders.get("map") is not None:
            prev = 6
            if(encoders["map"]["encoder"]["type"]=="conv"):
                modules = []
                conv_channels = encoders["map"]["encoder"]["channels"]
                for n in conv_channels:
                    modules.append(nn.Conv2d(prev, n, 3, padding=1))
                    modules.append(nn.ReLU())
                    prev = n
                self.encoders["map"] = nn.ModuleDict(
                    {
                        "encoder": nn.Sequential(*modules)
                    }
                )
            elif(encoders["map"]["encoder"]["type"]=="GeneralizedResNet"):
                self.encoders["map"] = nn.ModuleDict(
                    {
                        "encoder": build_backbone(encoders["map"]["encoder"]),
                    }
                )

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        # ego2global,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)
        # with torch.no_grad():
        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)
        num_cams = 6 # TODO hard-coded for 6 cameras
        bev_feat_list = []
        current_depth = None
        for i in range(0, N//num_cams): 
            # print("yyy")
            mlp_input = self.encoders["camera"]["vtransform"].get_mlp_input(camera2lidar, camera_intrinsics, post_rot=img_aug_matrix[..., :3, :3], post_tran=img_aug_matrix[..., :3, 3], bda=lidar_aug_matrix)
            with torch.set_grad_enabled(self.training and i == 0):
                bev_feat, depth = self.encoders["camera"]["vtransform"](
                    x[:,(num_cams * i):(num_cams * (i+1))],
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics[:,(num_cams * i):(num_cams * (i+1))],
                    camera2lidar[:,(num_cams * i):(num_cams * (i+1))],
                    img_aug_matrix[:,(num_cams * i):(num_cams * (i+1))],
                    lidar_aug_matrix,
                    img_metas,
                    mlp_input=mlp_input
                )
                if(i == 0):
                    current_depth = depth
                bev_feat_list.append(bev_feat)
        if(self.merge_bev_4d_cfg["type"] == "first"):
            merged_bev_feat = bev_feat_list[0]
        elif(self.merge_bev_4d_cfg["type"] == "concat"):
            merged_bev_feat = torch.cat(bev_feat_list, dim=1)
        return merged_bev_feat, current_depth

    def extract_lidar_features(self, x) -> torch.Tensor:
        # print("extr")
        # print(len(x))
        # print([p.shape for p in x])
        # print(x)
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        # with torch.no_grad():
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x
    #TODO
    def extract_map_features(self, x) -> torch.Tensor:
        x = self.encoders["map"]["encoder"](x)
        return x
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_depth=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                gt_depth,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        # feature_map=None,
        gt_depth=None,
        **kwargs,
    ):
        # print("sheap",img.shape, gt_depth.shape)
        features = []
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                with torch.set_grad_enabled(self.training and not self.freeze_state.get("camera")):
                    cam_feat, pred_depth = self.extract_camera_features(
                        img,
                        points,
                        camera2ego,
                        lidar2ego,
                        lidar2camera,
                        lidar2image,
                        camera_intrinsics,
                        camera2lidar,
                        img_aug_matrix,
                        lidar_aug_matrix,
                        metas,
                    )
                feature = cam_feat
            elif sensor == "lidar":
                with torch.set_grad_enabled(self.training and not self.freeze_state.get("lidar")):
                    lidar_feat = feature = self.extract_lidar_features(points)
                # print(points[0].detach().cpu().numpy().shape)
                # visualize_lidar("test_vis/l.png", points[0].detach().cpu().numpy())

                # print("z",feature.shape)
            elif sensor == "map":
                # canvas1 = np.transpose(gt_masks_bev[0,:3].detach().cpu().numpy(),(1,2,0))*255
                # canvas2 = np.transpose(feature_map[0,:3].detach().cpu().numpy(),(1,2,0))*255
                # fpath1 = "test_vis/f.png"
                # fpath2 = "test_vis/g.png"
                # mmcv.imwrite(canvas1, fpath1)
                # mmcv.imwrite(canvas2, fpath2)
                # print("x",gt_masks_bev.shape)
                map_feat = feature = self.extract_map_features(feature_map)[-1] # only use last element if resnet is used
                # print("y",feature.shape)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            # print(features[1][0].shape, features[1][-1].shape)
            # if(True):
            #     # TODO: hacky way of making map output work for res blocks
            #     features[1] = features[1][-1]
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]
        # print("xshape1", x.shape)
        x = self.decoder["backbone"](x)
        # print("xshape2", [elem.shape for elem in x])
        x = self.decoder["neck"](x)
        # print("xshape3", [elem.shape for elem in x])

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    # auxiliary loss
                    if "camera" in self.use_aux_loss:
                        aux_x = self.aux_loss_encoder([cam_feat])
                        aux_x = self.decoder["backbone"](aux_x)
                        aux_x = self.decoder["neck"](aux_x)
                        aux_pred_dict = head(aux_x, metas)
                        depth_preds = self.encoders["camera"]["vtransform"].depth_buffer
                        losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict, aux_preds_dicts=aux_pred_dict, depth_preds=depth_preds)
                    else:
                        losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                    
                    # print(self.loss_depth_weight)
                    if(self.loss_depth_weight > 0.0):
                        losses["depth"] = self.encoders["camera"]["vtransform"].get_depth_loss(gt_depth, pred_depth, self.loss_depth_weight)
                    
                        # TODO: Add 4D input and depth supervision
                        
                elif type == "map":
                    # if not torch.allclose(gt_masks_bev.float(),feature_map.float()):
                    #     print("was not equal!!!")
                        
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        # if "camera" in self.encoders.keys():
                        if self.visualize:
                            outputs[k].update({"pred_depth": pred_depth.cpu()})
                            # outputs[k].update({"pred_depth": self.encoders["camera"]["vtransform"].depth_buffer.cpu()})
                            outputs[k].update({"gt_depth": gt_depth.cpu()})
                            outputs[k].update({"cam_points": [cam_points.detach().cpu().numpy() for cam_points in self.encoders["camera"]["vtransform"].cam_points_buffer]})
                            self.encoders["camera"]["vtransform"].cam_points_buffer = []
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs
