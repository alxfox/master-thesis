# Script for running multiple different evaluations in a row
CONFIG_PATHS=('configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml' 'my_configs/baseline/nuscenes/det/centerhead/lssfpn/lidar+camera/bothcp/default.yaml' 'my_configs/baseline/nuscenes/det/centerhead/lssfpn/lidar+camera/bothcp/default.yaml')
CHKPT_PATHS=('data/runs/transfusionhead_fusion_baseline/run-e50d2cdf-2e16f888/' 'data/runs/centerhead_fusion_20cam_20lidar/no_cp/run-08334027-6bcee750/' 'data/runs/centerhead_fusion_bothcp_20_20/run-08334027-562b31c1/')

# CONFIG_PATH=
# CHKPT_PATH=
# OLDIFS=$IFS; IFS=',';
for i in {0..3..1} # first and last are inclusive
do
CONFIG_PATH=${CONFIG_PATHS[i]}

if [ -z $CONFIG_PATH ]; then
    echo "finished"
    break
fi;

echo ${CONFIG_PATHS[i]}
CHKPT_PATH=${CHKPT_PATHS[i]}'latest.pth'
OUT_PATH=${CHKPT_PATHS[i]}'eval_out.txt'

echo $OUT_PATH
echo $CHKPT_PATH
if [ -e $OUT_PATH ]
then 
echo "$OUT_PATH exists, skipping"
else
echo "$OUT_PATH doesn't exist, running evaluation"
torchpack dist-run -np 2 python tools/test.py $CONFIG_PATH $CHKPT_PATH --eval bbox | tail -n 22 >> $OUT_PATH
fi

for blackout in 'camera' 'lidar' 'fov120' 'fov180'
do
if [ $blackout == 'camera' ]; then OUT_PATH=${CHKPT_PATHS[i]}'eval_no_cam.txt' ; fi
if [ $blackout == 'lidar' ]; then OUT_PATH=${CHKPT_PATHS[i]}'eval_no_lidar.txt'; fi
if [ $blackout == 'fov120' ]; then OUT_PATH=${CHKPT_PATHS[i]}'eval_fov120.txt'; fi
if [ $blackout == 'fov180' ]; then OUT_PATH=${CHKPT_PATHS[i]}'eval_fov180.txt'; fi

if [ -e $OUT_PATH ]
then
echo "$OUT_PATH exists, skipping"
else
echo "$OUT_PATH doesn't exist, running evaluation"
torchpack dist-run -np 2 python tools/test.py $CONFIG_PATH $CHKPT_PATH --blackout $blackout --eval bbox | tail -n 22 >> $OUT_PATH
fi
done
done
