#########################################################################
# clone CUDA 8.0 from NVIDIA
#########################################################################
# get driver
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

#########################################################################
# install all the packges needed by CUDA itself
# (only once: if you install a second fork of cuda you do not need to install it twice)
#########################################################################

source ~/DNNDK_Installation/scripts/caffe/install_cuda_packages.sh

#install cuda 9.0
sudo apt-get install cuda-9-0
sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 2505,875 # performance optimziation from google suggestion

# add to environment variable
echo '# NVIDIA CUDA Toolkit' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64' >> ~/.bashrc
source ~/.bashrc
nvidia-smi

#test cuda 9.0
cd /usr/local/cuda-9.0/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
cd ~
#########################################################################
# clone CuDNN 7.0.5 from NVIDIA
########################################################################

wget http://developer.download.nvidia.com/compute/redist/cudnn/v7.0.5/cudnn-9.0-linux-x64-v7.tgz

#########################################################################
# install all the packges needed by CUDNN 7.0.5 itself
# (only once: if you install a second fork of cudnn you do not need to install it twice)
#########################################################################

tar xzvf cudnn-9.0-linux-x64-v7.tgz
cd cuda
sudo cp -P ./lib64/* /usr/local/cuda/lib64/
sudo cp -P ./include/* /usr/local/cuda/include/
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

cd ~

#########################################################################
# clone NCCL from NVIDIA github
########################################################################

#git clone https://github.com/NVIDIA/nccl.git
wget https://forums.xilinx.com/xlnx/attachments/xlnx/Deephi/60/1/nccl1.tar.gz

#########################################################################
# install all the packges needed by NCCL itself
# (only once: if you install a second fork of cudnn you do not need to install it twice)
#########################################################################

tar zxvf nccl1.tar.gz
cd nccl-master
sudo make CUDA_HOME=/usr/local/cuda test
sudo make CUDA_HOME=/usr/local/cuda install
cd ~

# add to environment variable
echo '# NCCL Library' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib' >> ~/.bashrc
echo "Done to NVIDIA Installation"
