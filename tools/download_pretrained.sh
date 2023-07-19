mkdir pretrained &&
cd pretrained &&
wget https://bevfusion.mit.edu/files/pretrained_updated/bevfusion-det.pth --no-check-certificate &&
wget https://bevfusion.mit.edu/files/pretrained_updated/bevfusion-seg.pth --no-check-certificate &&
wget https://bevfusion.mit.edu/files/pretrained/lidar-only-det.pth --no-check-certificate &&
wget https://bevfusion.mit.edu/files/pretrained/lidar-only-seg.pth --no-check-certificate &&
wget https://bevfusion.mit.edu/files/pretrained_updated/camera-only-det.pth --no-check-certificate &&
wget https://bevfusion.mit.edu/files/pretrained_updated/camera-only-seg.pth --no-check-certificate &&
wget https://bevfusion.mit.edu/files/pretrained_updated/swint-nuimages-pretrained.pth --no-check-certificate

