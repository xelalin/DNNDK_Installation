# packages needed by BVLC CAFFE 1.0
# execute from Python2.7 virtualenv

sudo apt-get update
sudo apt-get upgrade

sudo apt-get install -y --force-yes build-essential cmake git unzip pkg-config
sudo apt-get install -y --force-yes libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install -y --force-yes libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y --force-yes libxvidcore-dev libx624-dev
sudo apt-get install -y --force-yes libxvidcore-dev libx264-dev
sudo apt-get install -y --force-yes libgtk-3-dev
sudo apt-get install -y --force-yes libhdf5-serial-dev graphviz
sudo apt-get install -y --force-yes libopenblas-dev libatlas-base-dev gfortran
sudo apt-get install -y --force-yes python2.7-dev python3-dev
sudo apt-get install -y --force-yes linux-image-generic linux-image-extra-virtual
sudo apt-get install -y --force-yes linux-source linux-headers-generic
sudo apt-get install -y --force-yes libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev
sudo apt-get install -y --no-install-recommends libboost-all-dev

sudo apt-get install -y --force-yes build-essential
sudo apt-get install -y --force-yes cmake
sudo apt-get install -y --force-yes git
sudo apt-get install -y --force-yes pkg-config
sudo apt-get install -y --force-yes libprotobuf-dev
sudo apt-get install -y --force-yes libleveldb-dev
sudo apt-get install -y --force-yes libsnappy-dev
sudo apt-get install -y --force-yes libhdf5-serial-dev
sudo apt-get install -y --force-yes protobuf-compiler
sudo apt-get install -y --force-yes libatlas-base-dev
sudo apt-get install -y --force-yes libgflags-dev
sudo apt-get install -y --force-yes libgoogle-glog-dev
sudo apt-get install -y --force-yes liblmdb-dev
sudo apt-get install -y --force-yes python-pip
sudo apt-get install -y --force-yes python-dev
sudo apt-get install -y --force-yes python-numpy
sudo apt-get install -y --force-yes python-scipy
sudo apt-get install -y --force-yes python-opencv
sudo apt-get install -y --force-yes python-lmdb
sudo apt-get install -y --force-yes libopencv-dev

#cp ~/scripts/caffe/test_layer_factory.cpp  $CAFFE_ROOT/src/caffe/test/test_layer_factory.cpp 
