from typing import Tuple

from mmcv.runner import force_fp32
from torch import nn
import torch.nn.functional as F
import torch
from mmdet3d.models.builder import VTRANSFORMS
from torch.cuda.amp.autocast_mode import autocast
from .base import BaseTransform

__all__ = ["LSSTransform"]


@VTRANSFORMS.register_module()
class LSSTransform(BaseTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
        depth_decay = False
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
            depth_decay=depth_decay
        )
        self.depthnet = nn.Conv2d(in_channels, self.D + self.C, 1)
        self.downsample_ratio = 8
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x):
        # The input image (features) are fed through the depth network.
        # The results is a tensor with depth predictions for each pixel as well as the final features. 
        # These two parts are then merged such that we have a tensor that gives a set of C features for every image feature pixel at every depth (at D depths).
        # print(x.shape)
        B, N, C, fH, fW = x.shape
        # baaa torch.Size([8, 6, 256, 32, 88]) 34603008
        # caaa torch.Size([48, 198, 32, 88]) 26763264
        # daaa torch.Size([48, 80, 118, 32, 88]) 1275985920
        # eaaa torch.Size([8, 6, 118, 32, 88, 80]) 1275985920
        # print("baaa",x.shape, x.nelement())
        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x)
        # print("caaa",x.shape, x.nelement())
        depth = x[:, : self.D].softmax(dim=1)
        self.depth_buffer = depth.unsqueeze(-1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)
        # print("daaa",x.shape, x.nelement())

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        # print("eaaa",x.shape, x.nelement())
        return x, depth

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.downsample_ratio,
                                   self.downsample_ratio, W // self.downsample_ratio,
                                   self.downsample_ratio, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample_ratio * self.downsample_ratio)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_ratio,
                                   W // self.downsample_ratio)

        # if not self.sid:
        gt_depths = (gt_depths - (self.dbound[0] -
                                    self.dbound[2])) / \
                    self.dbound[2]
        # else:
        #     gt_depths = torch.log(gt_depths) - torch.log(
        #         torch.tensor(self.grid_config['depth'][0]).float())
        #     gt_depths = gt_depths * (self.D - 1) / torch.log(
        #         torch.tensor(self.grid_config['depth'][1] - 1.).float() /
        #         self.grid_config['depth'][0])
        #     gt_depths = gt_depths + 1.
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds, loss_depth_weight):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return loss_depth_weight * depth_loss

    def forward(self, *args, **kwargs):
        x, depth = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x, depth
