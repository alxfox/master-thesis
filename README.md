# BEVFusion
torchpack dist-run -np 1 python tools/train.py idp_configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --checkpoint pretrained/bevfusion-det.pth --mode pred 
torchpack dist-run -np 1 python tools/visualize.py idp_configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml