# Robust BEVFusion

This is a modified version of the [BEVFusion](https://github.com/mit-han-lab/bevfusion) framework.

Key changes are:

- All models run using the CenterHead model. Only the object detection task is supported. The performance is worse than the TransFusionHead but training is stable.
    
    - The TransFusionHead is very unstable during training if hyperparameters are not correctly tuned (and will crash in that case) making it hard to use for experimentation

- All used configurations for experiments may be found under ```configs/nuscenes/det/centerhead/lssfpn/```
- It is recommended to always load config files from the config folder instead of from the copied version next to the checkpoint


## Usage

#### Fusion Training
```
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from pretrained/lidar-only-det.pth 
```

#### Camera Training
```
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

#### LiDAR Training
```
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml
```


## Setting up the Environment
### 1. Create Docker Image
```
$ cd docker && docker build . -t bevfusion-robust
```

### 2. Create Docker Container
```
$ nvidia-docker run -it -v `pwd`/../data:/dataset --shm-size 16g bevfusion-robust /bin/bash

$ docker attach <name>

Options:
--shm-size 16g (shared memory size)
-v `pwd`/../data:/dataset (bind host folder to docker)
------------
-p 16006:6006 (bind port for tensorboard)
```

**if nvidia-docker is not installed:**

```
Replace:
$ nvidia-docker run ...

With:
$ docker run --runtime=nvidia --gpus all ...
```
### 3. Ensure versions are correct
There is an issue with the mmcv install so run this:
```
$ pip uninstall mmcv
$ pip uninstall mmcv-full

$ MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install mmcv==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
$ MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```
If one of these commands fails, just ignore it (mmcv-full was renamed to mmcv at some point so there is something funky going on with the versions)
Afterwards run the following to ensure all versions are correct:
```
$ pip install setuptools==59.5.0
$ pip install numpy==1.23.5
$ pip install yapf==0.40.1
```
### 4. Install the repository in the docker
```
$ cd home && git clone https://gitlab.lrz.de/perception/bevfusion.git
$ cd bevfusion && git checkout robustness
$ ln -s /dataset data
$ python setup.py develop
```
**Folder structure should look like this:**
```
bevfusion
├── ...
├── tools
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```

### 5. Run Dataset Preprocessing (Once)
```
$ python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```
Note: This command seems to imply that you can have the preprocessed files in a different folder from the dataset. You **cannot**. There are sections in the code that assume they are in the same folder.