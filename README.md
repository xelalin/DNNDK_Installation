Installing Xilinx ML tools(Caffe 1.0 BVLC, Caffe Ristretto DNNDK and Tnesorflow) for Ubuntu 16.04 Linux on GCP(Google Cloud Platform)
======================================================================================

This document tries to explain how installing ML SW environments as Caffe and TensorFlow/Keras on GCP. Instead of Caffe you will install Ristretto. The dependencies of Caffe on a lot of Linux packages can make the install and compilation processes very painful and time consuming.

The DNNDK requires the Host (PC) side of its ML toolchain to be compatible with:
- 	Ubuntu 16.04 + CuDA 8.0 + CuDNN v7.0.5
- 	Ubuntu 16.04 + CuDA 9.0 + CuDNN v7.0.5
- 	Ubuntu 16.04 + CuDA 9.1 + CuDNN v7.0.5

Caffe 1.0 BVLC and Caffe-Ristretto require:
-	Python 2.7

TensorFlow-GPU 1.5.0 requires:
-	either Python 2.7 or Python 3.5
-	CuDA 9.0

Others Tools require:
-	OpenCV 2.x

In my case I have selected CuDA 9.0 because I could have both TensorFlow and Caffe in the same virtual environment. please follow the process to install drivers and tools step by step.

1.  Create Project and GPU Instance on GCP
2.  SSH to GCP Instance
3.  Install CUDA-9.0 Toolkits
4.  Install CuDNN 7.0.5
5.  Install NCCL (optional for DNNDK v2.08)
6.  Create Python2.7 Virtual Env. with ML-related package
7.  Install Caffe-Ristretto
8.  Install DNNDK
9.  Build and Delopy Network Model(Resnet50)
10. Quickly Installation by scripts

About TensorFlow need which version CuDA toolkits, please refer to [here](https://www.tensorflow.org/install/source#tested_source_configurations)
**NOTES: The article is based on Deephi DNNDK v2.08 beta version** 

# Create Project and GPU Instance on GCP
Here will be showing you how to create a basic virtual machine in Google Cloud Platform.
## Prerequisites
Google account with billing enabled(If you are doing it for the first time just create a billing account using any credit/debit card with international transaction enables and you will get free credits from Google), Please refer to [here](https://cloud.google.com/resource-manager/docs/creating-managing-projects) about Creating and Managing Projects on GCP.
## Create Virtual Machine with GPU Instance
Below environment settings are for me, you could adjust it based on your requiremets
-	Zone : asia-east1-b
-	Machine type : 4 x vCPU, 1 x NVIDIA Telsa K80 GPU, 15GB Memory
-	Boot disk : Ubuntu 16.04 LTS, Size(GB) : 128
-	Enable Allow HTTP traffic and Allow HTTPS traffic

For detail, please refer to [here](https://cloud.google.com/compute/docs/quickstart-linux)

# SSH to GCP Instance
You can access your created VM by web console. if you don't habit the inferface,  you can install Google cloud SDK then use gcloud instruction to connect the VM through SSH. The instruction is like the ssh of Linux:

    gcloud compute ssh [INSTANCE_NAME]


In my case, I input below command in my local Linux machine ternimal

    gcloud compute --project "xelalin-218015" ssh --zone "asia-east1-b" instance-1

For details, please refer to [here](https://cloud.google.com/sdk/docs/quickstart-linux)

# Install CUDA-9.0 Tookits
Clone CUDA ToolKits from NVIDIA Website with the following command:

    curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

Run the following command in GCP console to install the required libraries:

    sudo apt-get update
    sudo apt-get upgrade
    sudo apt-get install -y --force-yes build-essential autoconf libtool libopenblas-dev libgflags-dev
    sudo apt-get install -y --force-yes libgoogle-glog-dev libopencv-dev libprotobuf-dev protobuf-compiler
    sudo apt-get install -y --force-yes libleveldb-dev liblmdb-dev
    sudo apt-get install -y --force-yes libhdf5-dev libsnappy-dev libboost-system-dev libboost-thread-dev libboost-filesystem-dev
    sudo apt-get install -y --force-yes libyaml-cpp-dev libssl-dev
    sudo apt-get install -y --force-yes build-essential cmake git unzip pkg-config
    sudo apt-get install -y --force-yes libjpeg-dev libtiff5-dev libjasper-dev
    sudo apt-get install -y --force-yes libpng12-dev
    sudo apt-get install -y --force-yes libavcodec-dev libavformat-dev libswscale-dev
    sudo apt-get install -y --force-yes libv4l-dev
    sudo apt-get install -y --force-yes libxvidcore-dev libx264-dev
    sudo apt-get install -y --force-yes libgtk-3-dev
    sudo apt-get install -y --force-yes libhdf5-serial-dev graphviz
    sudo apt-get install -y --force-yes libopenblas-dev libatlas-base-dev gfortran
    sudo apt-get install -y --force-yes python-tk python3-tk python-imaging-tk
    sudo apt-get install -y --force-yes python2.7-dev python3-dev
    sudo apt-get install -y --force-yes linux-image-generic linux-image-extra-virtual
    sudo apt-get install -y --force-yes linux-source linux-headers-generic

Update to CUDA 9.0

    sudo apt-get update
    sudo apt-get install cuda-9-0

Enable and test NVIDIA GPU

    sudo nvidia-smi -pm 1
    sudo nvidia-smi -ac 2505,875 # performance optimziation from google suggestion

Add to CUDA ToolKits environment variable

    echo '# NVIDIA CUDA Toolkit' >> ~/.bashrc
    echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
    echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64' >> ~/.bashrc

Verify successfull driver installation

    source ~/.bashrc
    nvidia-smi

The output should looks like:
```console
Tue Dec 11 01:29:36 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.79       Driver Version: 410.79       CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K80           On   | 00000000:00:04.0 Off |                    0 |
| N/A   33C    P8    31W / 149W |      0MiB / 11441MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
Test CUDA toolkits

    cd /usr/local/cuda-9.0/samples/1_Utilities/deviceQuery
    sudo make
    ./deviceQuery
    cd ~
    
The output should looks like:
```console
./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "Tesla K80"
  CUDA Driver Version / Runtime Version          10.0 / 9.0
  CUDA Capability Major/Minor version number:    3.7
  Total amount of global memory:                 11441 MBytes (11996954624 bytes)
  (13) Multiprocessors, (192) CUDA Cores/MP:     2496 CUDA Cores
  GPU Max Clock rate:                            824 MHz (0.82 GHz)
  Memory Clock rate:                             2505 Mhz
  Memory Bus Width:                              384-bit
  L2 Cache Size:                                 1572864 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
  Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Supports Cooperative Kernel Launch:            No
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 0 / 4
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 10.0, CUDA Runtime Version = 9.0, NumDevs = 1
Result = PASS
```
# Install CuDNN 7.0.5
Clone CuDNN 7.0.5 from NVIDIA Website with following command:

    wget http://developer.download.nvidia.com/compute/redist/cudnn/v7.0.5/cudnn-8.0-linux-x64-v7.tgz

Install all the packges needed by CuDNN 7.0.5 itself (only once: if you install a second fork of CuDNN you do not need to install it twice)

    tar xzvf cudnn-9.0-linux-x64-v7.tgz
    cd cuda
    sudo cp -P ./lib64/* /usr/local/cuda/lib64/
    sudo cp -P ./include/* /usr/local/cuda/include/
    sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
    cd ~

# Install NCCL1
The DECENT compression tool requires **libnccl.so.1** NVIDIA library which is no more supported, in favor of most recent nccl 2.x. Go to the following to download the NCCL from Xilinx forum

    wget https://forums.xilinx.com/xlnx/attachments/xlnx/Deephi/60/1/nccl1.tar.gz
    tar zxvf nccl1.tar.gz
    cd nccl-master
    sudo make CUDA_HOME=/usr/local/cuda test
    
Test binaries are located in the subdirectories nccl/build/test/{single,mpi}.

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib
    ./build/test/single/all_reduce_test 10000000

The output should looks like:
```console
# Using devices
#   Device  0 ->  0 [0x0a] GeForce GTX TITAN X
#   Device  1 ->  1 [0x09] GeForce GTX TITAN X
#   Device  2 ->  2 [0x06] GeForce GTX TITAN X
#   Device  3 ->  3 [0x05] GeForce GTX TITAN X

#                                                 out-of-place                    in-place
#      bytes             N    type      op     time  algbw  busbw      res     time  algbw  busbw      res
    10000000      10000000    char     sum    1.628   6.14   9.21    0e+00    1.932   5.18   7.77    0e+00
    10000000      10000000    char    prod    1.629   6.14   9.21    0e+00    1.643   6.09   9.13    0e+00
    10000000      10000000    char     max    1.621   6.17   9.25    0e+00    1.634   6.12   9.18    0e+00
    10000000      10000000    char     min    1.633   6.12   9.19    0e+00    1.637   6.11   9.17    0e+00
    10000000       2500000     int     sum    1.611   6.21   9.31    0e+00    1.626   6.15   9.23    0e+00
    10000000       2500000     int    prod    1.613   6.20   9.30    0e+00    1.629   6.14   9.21    0e+00
    10000000       2500000     int     max    1.619   6.18   9.26    0e+00    1.627   6.15   9.22    0e+00
    10000000       2500000     int     min    1.619   6.18   9.27    0e+00    1.624   6.16   9.24    0e+00
    10000000       5000000    half     sum    1.617   6.18   9.28    4e-03    1.636   6.11   9.17    4e-03
    10000000       5000000    half    prod    1.618   6.18   9.27    1e-03    1.657   6.03   9.05    1e-03
    10000000       5000000    half     max    1.608   6.22   9.33    0e+00    1.621   6.17   9.25    0e+00
    10000000       5000000    half     min    1.610   6.21   9.32    0e+00    1.627   6.15   9.22    0e+00
    10000000       2500000   float     sum    1.618   6.18   9.27    5e-07    1.622   6.17   9.25    5e-07
    10000000       2500000   float    prod    1.614   6.20   9.29    1e-07    1.628   6.14   9.21    1e-07
    10000000       2500000   float     max    1.616   6.19   9.28    0e+00    1.633   6.12   9.19    0e+00
    10000000       2500000   float     min    1.613   6.20   9.30    0e+00    1.628   6.14   9.21    0e+00
    10000000       1250000  double     sum    1.629   6.14   9.21    0e+00    1.628   6.14   9.21    0e+00
    10000000       1250000  double    prod    1.619   6.18   9.26    2e-16    1.628   6.14   9.21    2e-16
    10000000       1250000  double     max    1.613   6.20   9.30    0e+00    1.630   6.13   9.20    0e+00
    10000000       1250000  double     min    1.622   6.16   9.25    0e+00    1.623   6.16   9.24    0e+00
```
To install, run `make PREFIX=<install dir> install` and add `<instal dir>/lib` to your `LD_LIBRARY_PATH`.

    sudo make CUDA_HOME=/usr/local/cuda install
    
Add to NCCL library environment variable

    echo '# NCCL Library' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib' >> ~/.bashrc


# Create Python2.7 Virtual Env. with ML-related package
Perform all the following install instructions to setup Python Virtual Env.:

    wget https://bootstrap.pypa.io/get-pip.py
    sudo python get-pip.py
    sudo python3 get-pip.py
    sudo pip2 install virtualenv virtualenvwrapper
    sudo rm -rf ~/.cache/pip get-pip.py 

Add the following lines to the ~/.bashrc file

    echo '# Virtualenv and virtualenvwrapper' >> ~/.bashrc
    echo 'export WORKON_HOME=$HOME/.virtualenvs' >> ~/.bashrc
    echo 'export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python2' >> ~/.bashrc
    echo 'export VIRTUALENVWRAPPER_VIRTUALENV=/usr/local/bin/virtualenv' >> ~/.bashrc
    echo 'source /usr/local/bin/virtualenvwrapper.sh' >> ~/.bashrc
    echo 'export PYTHONPATH=/usr/local/lib/python2.7/dist-packages:$PYTHONPATH' >> ~/.bashrc

Now reload the changes by running:

    source ~/.bashrc

Create a Python2, based virtual environment, for example **"caffe_py27"**:

    mkvirtualenv caffe_py27 -p python2
    workon caffe_py27
    
Import all the following ML related packages by running the commands(note that you must be inside the virtualenv)

    pip2 install absl-py==0.1.13
    pip2 install astor==0.6.2
    pip2 install backports.functools-lru-cache==1.5
    pip2 install backports.shutil-get-terminal-size==1.0.0
    pip2 install backports.weakref==1.0.post1
    pip2 install bleach==1.5.0
    pip2 install certifi==2018.1.18
    pip2 install chardet==3.0.4
    pip2 install click==6.7
    pip2 install cloudpickle==0.5.3
    pip2 install cycler==0.10.0
    pip2 install Cython==0.28.4
    pip2 install dask==0.18.2
    pip2 install decorator==4.3.0
    pip2 install enum34==1.1.6
    pip2 install funcsigs==1.0.2
    pip2 install futures==3.2.0
    pip2 install gast==0.2.0
    pip2 install graphviz==0.8.4
    pip2 install grpcio==1.10.1
    pip2 install h5py==2.8.0
    pip2 install html5lib==0.9999999
    pip2 install idna==2.6
    pip2 install imutils==0.4.6
    pip2 install ipython==5.7.0
    pip2 install ipython-genutils==0.2.0
    pip2 install Keras==2.1.5
    pip2 install kiwisolver==1.0.1
    pip2 install leveldb==0.194
    pip2 install lmdb==0.94
    pip2 install Markdown==2.6.11
    pip2 install matplotlib==2.2.2
    pip2 install mock==2.0.0
    pip2 install networkx==2.1
    pip2 install nose==1.3.7
    pip2 install numpy==1.15.0
    pip2 install olefile==0.44
    pip2 install pandas==0.23.3
    pip2 install pathlib2==2.3.2
    pip2 install pbr==4.2.0
    pip2 install pexpect==4.6.0
    pip2 install pickleshare==0.7.4
    pip2 install Pillow==5.2.0
    pip2 install progressbar2==3.37.0
    pip2 install prompt-toolkit==1.0.15
    pip2 install protobuf==3.6.0
    pip2 install ptyprocess==0.6.0
    pip2 install pydot==1.2.4
    pip2 install Pygments==2.2.0
    pip2 install pyparsing==2.2.0
    pip2 install python-dateutil==2.5.0
    pip2 install python-gflags==3.1.2
    pip2 install python-utils==2.3.0
    pip2 install pytz==2018.5
    pip2 install PyWavelets==0.5.2
    pip2 install PyYAML==3.13
    pip2 install requests==2.18.4
    pip2 install scandir==1.7
    pip2 install scikit-image==0.14.0
    pip2 install scikit-learn==0.19.1
    pip2 install scipy==1.0.0
    pip2 install simplegeneric==0.8.1
    pip2 install six==1.11.0
    pip2 install stevedore==1.29.0
    pip2 install subprocess32==3.5.2
    pip2 install tensorboard==1.7.0
    pip2 install tensorflow-gpu==1.5.0
    pip2 install tensorflow-tensorboard==1.5.1
    pip2 install termcolor==1.1.0
    pip2 install toolz==0.9.0
    pip2 install traitlets==4.3.2
    pip2 install urllib3==1.22
    pip2 install virtualenv==16.0.0
    pip2 install virtualenv-clone==0.3.0
    pip2 install virtualenvwrapper==4.8.2
    pip2 install wcwidth==0.1.7
    pip2 install Werkzeug==0.14.1

Install other packages needed by Caffe, included OpenCV 2.4
Exit Python2 virtualenv

    deactivate

Follow the next instructions out of any python virtualenv:

    sudo apt-get update
    sudo apt-get upgrade
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

Note that the below packages should be already installed from previous sections

    sudo apt-get install -y --force-yes build-essential cmake git unzip pkg-config
    sudo apt-get install -y --force-yes libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
    sudo apt-get install -y --force-yes libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
    sudo apt-get install -y --force-yes libxvidcore-dev libx624-dev
    sudo apt-get install -y --force-yes libgtk-3-dev
    sudo apt-get install -y --force-yes libhdf5-serial-dev graphviz
    sudo apt-get install -y --force-yes libopenblas-dev libatlas-base-dev gfortran
    sudo apt-get install -y --force-yes python2.7-dev python3-dev
    sudo apt-get install -y --force-yes linux-image-generic linux-image-extra-virtual
    sudo apt-get install -y --force-yes linux-source linux-headers-generic
    sudo apt-get install -y --force-yes libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev
    sudo apt-get install -y --no-install-recommends libboost-all-dev

For OpenCV 2.4, You only need to run the below command:

    sudo apt-get install -y libopencv-dev

Note that the cv2 package was installed in `/usr/lib/python2.7/dist-packages` instead of `/usr/local/lib/python2.7/dist-packages`, therefore I applied the following workaround:

    cd ~/.virtualenvs/caffe_py27/local/lib/python2.7/site-packages/
    ln -s /usr/lib/python2.7/dist-packages/cv2.x86_64-linux-gnu.so cv2.so
    cd ~

Test your installed library, please entry python virtualenv with the following command:

    workon caffe_py27

Test OpenCV installation, whatever python you are using

    python
    import cv2 as cv
    cv.__version__
    cv.__file__
    quit()

In case cv2 would not be import(for python2 for example), just add the following 2 lines just before the "import cv2":

    python
    import sys
    sys.path.append("/usr/local/lib/python2.7/site-packages")
    import cv2 as cv
    cv.__version__
    cv.__file__
    quit()

Test your TensorFlow installation:

    python
    import tensorflow as tf
    tf.__version__
    quit()

If you have errors after <import panda>, please re-install panda in python virtualenv with the following command:

    sudo pip2 install pandas

Test your Keras installation:

    python
    import keras as ks
    ks.__version__
    quit()

# Install Caffe-Ristrettorst 
Ristretto is public available [here](https://github.com/pmgysel/caffe).It was developed by GL-Research. There is a WiKi page associated to it. [Ristretto Wiki](https://gl-research.com/caffe/ristretto/wikis/home) and you need to contact the owner to get access to it (Javier Garcia, [email address](jgarcia@gl-research.com).
First of all you need to download the code with following command:

    git clone https://github.com/pmgysel/caffe

Move Ristretto Caffe to caffe_tools folder

    mkdir ~/caffe_tools
    mv ~/caffe ~/caffe_tools/Ristretto
    cd ~

Modify Makefile.config and Make to suite your Env with following commands:

    export CAFFE_ROOT=$HOME/caffe_tools/Ristretto
    workon caffe_py27
    cd $CAFFE_ROOT
    cp Makefile.config.example Makefile.config

Now edit **Makefile.config** before running any other step with vi Makefile.config
-	Uncomment the `CPU_ONLY:=1` line at the top, if you don't have a GPU. On the contrary, if you have a GPU, leave it commented and uncomment `USE_CUDNN:=1`
-	`USE_OPENCV:=1`, to enable OpenCV.
-	Remove the following 2 lines
>-gencode arch=compute_20,code=sm_20 \
>-gencode arch=compute_20,code=sm_21 \
-	Check your PYTHON_INCLUDE to be similar to this (add the third line only if your python2.7 previous installation generated files in both `/usr/lib` and `/usr/local/lib`, otherwise leave it commented)
> PYTHON_INCLUDE := /usr/include/python2.7
>                  /usr/lib/python2.7/dist-packages/numpy/core/include
>                 /usr/local/lib/python2.7/dist-packages/numpy/core/include
-	Add the following two lines to Makefile.config (make sure each line stays in a single line, without any carriage return), otherwise you will get a compilation error during the step of make all

> INCLUDE_DIRS:=$(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial \
> LIBRARY_DIRS:=$(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial /usr/local/share/OpenCV/3rdparty/lib
-	Now you should edit also the Makefile, to avoid possible future compilation errors that otherwise could compare. You can first do a trial without modifying it and then, if you got errors, try the below changes:
>1. Replace the following line: \
>```NVCCFLAGS+= -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)``` \
>with the following line (everything on the same line, no carriage-return)\
>```NVCCFLAGS+=-D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)``` \
>2. Now open the file **CMakeLists.txt** and add the following two lines:\
>```# ---[ Includes```\
>```set(${CMAKE_CXX_FLAGS} "-D_FORCE_INLINES ${CMAKE_CXX_FLAGS}")```
-	Now you can run the following commands to build and test Caffe-SSD-Ristretto (note that 8 is the amount of parallel CPUs of your PC, if you have less you need to use a small number):
```shell
cd /usr/lib/x86_64-linux-gnu
sudo ln -s libhdf5_serial.so.10.0.2 libhdf5.so
sudo ln -s libhdf5_serial_hl.so.10.0.2 libhdf5_hl.so
cd $CAFFE_ROOT
make clean
make all  -j8
make test -j8
make pycaffe
make distribute
make runtest -j8
```
If there are some errors on make runtest as `unknown file: Failure C++ exception with description "locale::facet::_S_create_c_locale name not valid" thrown in the test body.`, please add the following command:

    export LC_ALL="C"

The output should looks like:
```console
[ RUN      ] LSTMLayerTest/2.TestLSTMUnitGradient
[       OK ] LSTMLayerTest/2.TestLSTMUnitGradient (478 ms)
[ RUN      ] LSTMLayerTest/2.TestLSTMUnitGradientNonZeroCont
[       OK ] LSTMLayerTest/2.TestLSTMUnitGradientNonZeroCont (495 ms)
[ RUN      ] LSTMLayerTest/2.TestForward
[       OK ] LSTMLayerTest/2.TestForward (11 ms)
[ RUN      ] LSTMLayerTest/2.TestGradientNonZeroCont
[       OK ] LSTMLayerTest/2.TestGradientNonZeroCont (2943 ms)
[ RUN      ] LSTMLayerTest/2.TestGradientNonZeroContBufferSize2
[       OK ] LSTMLayerTest/2.TestGradientNonZeroContBufferSize2 (6869 ms)
[ RUN      ] LSTMLayerTest/2.TestGradient
[       OK ] LSTMLayerTest/2.TestGradient (2953 ms)
[----------] 9 tests from LSTMLayerTest/2 (49409 ms total)

[----------] 10 tests from ConcatLayerTest/0, where TypeParam = caffe::CPUDevice<float>
[ RUN      ] ConcatLayerTest/0.TestSetupChannels
[       OK ] ConcatLayerTest/0.TestSetupChannels (0 ms)
[ RUN      ] ConcatLayerTest/0.TestSetupChannelsNegativeIndexing
[       OK ] ConcatLayerTest/0.TestSetupChannelsNegativeIndexing (1 ms)
[ RUN      ] ConcatLayerTest/0.TestGradientChannelsBottomOneOnly
[       OK ] ConcatLayerTest/0.TestGradientChannelsBottomOneOnly (2 ms)
[ RUN      ] ConcatLayerTest/0.TestForwardChannels
[       OK ] ConcatLayerTest/0.TestForwardChannels (0 ms)
[ RUN      ] ConcatLayerTest/0.TestForwardTrivial
[       OK ] ConcatLayerTest/0.TestForwardTrivial (0 ms)
[ RUN      ] ConcatLayerTest/0.TestGradientNum
[       OK ] ConcatLayerTest/0.TestGradientNum (4 ms)
[ RUN      ] ConcatLayerTest/0.TestGradientChannels
[       OK ] ConcatLayerTest/0.TestGradientChannels (3 ms)
[ RUN      ] ConcatLayerTest/0.TestSetupNum
[       OK ] ConcatLayerTest/0.TestSetupNum (0 ms)
[ RUN      ] ConcatLayerTest/0.TestForwardNum
[       OK ] ConcatLayerTest/0.TestForwardNum (0 ms)
[ RUN      ] ConcatLayerTest/0.TestGradientTrivial
[       OK ] ConcatLayerTest/0.TestGradientTrivial (2 ms)
[----------] 10 tests from ConcatLayerTest/0 (12 ms total)

[----------] Global test environment tear-down
[==========] 2059 tests from 272 test cases ran. (743141 ms total)
[  PASSED  ] 2059 tests.
```
Add the following lines to the ~/.bashrc file

    echo '# MACHINE LEARNING' >> ~/.bashrc
    echo 'export CAFFE_ROOT=$HOME/caffe_tools/Ristretto' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$CAFFE_ROOT/distribute/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo 'export PYTHONPATH=$CAFFE_ROOT/distribute/python:$PYTHONPATH' >> ~/.bashrc
    source ~/.bashrc

Test Caffe and Tensorflow Env.

    echo $CAFFE_ROOT
    echo $LD_LIBRARY_PATH
    echo $PYTHONPATH
    python ~/DNNDK_Installation/scripts/caffe/test_caffe.py

# Install DNNDK
DeePhi's tools for ML implementation on Xilinx FPGA called DNNDK that are split in two parts, one running on the Ubuntu 16.04 PC(the Host) and another running on the board(the Target).Download Deephi tools and board image files form the DeePhi website - http://deephi.com/technology/dnndk

Board Images:
-	[DP-8020](http://deephi.com/assets/2018-12-04-dp-8020-desktop-stretch.img.zip)
-	[DP-N1](http://deephi.com/assets/2018-12-12-dp-n1-desktop-stretch.img.zip)
-	[Ultra96](http://deephi.com/assets/xilinx-ultra96-desktop-stretch-2018-12-10.img.zip)	
-	[ZCU102](http://www.deephi.com/assets/2018-12-04-zcu102-desktop-stretch.img.zip)
-	[ZCU104](http://www.deephi.com/assets/2018-12-04-zcu104-desktop-stretch.img.zip)

If you download it at local machine, please follow below command to upload the DNNDK compressed archive file to GCP cloud Instance and Keep Board Images on local machine

    gcloud compute scp [[USER@]INSTANCE:]SRC [[[USER@]INSTANCE:]SRC …] [[USER@]INSTANCE:]DEST [--compress] [--dry-run] [--force-key-file-overwrite] [--plain] [--port=PORT] [--recurse] [--scp-flag=SCP_FLAG] [--ssh-key-file=SSH_KEY_FILE] [--strict-host-key-checking=STRICT_HOST_KEY_CHECKING] [--zone=ZONE] [GCLOUD_WIDE_FLAG …]

More detals, see [Google Cloud SDK Doc](https://cloud.google.com/sdk/gcloud/reference/compute/scp)
Example:

    gcloud compute scp deephi_dnndk_v2.08_beta.tar.gz instance-1:/home/alex/ML

When done to upload the compressed archive file, please ssh to GCP Instance then follow the command:

    tar zxvf deephi_dnndk_v2.08_beta.tar.gz
    cd deephi_dnndk_v2.08_beta
    cd host_x86
    sudo ./install.sh ZCU102
    cd /usr/local/bin
    ./dnnc
    ./decent

If you could finally see following output, that means you have completed all the install process successfully!!
```console
decent: command line brew
usage: fix_tool <command> <args>

commands:
  quantize        transform float model to fix (need calibration with dataset) and deploy to DPU
  deploy          deploy finetuned model to DPU
  finetune        train or finetune a model
  test            score a model
example:
  1. quantize:                      ./decent_q quantize -model float.prototxt -weights float.caffemodel -gpu 0
  2. quantize with auto test:       ./decent_q quantize -model float.prototxt -weights float.caffemodel -gpu 0 -auto_test -test_iter 50
  3. quantize with method 0:        ./decent_q quantize -model float.prototxt -weights float.caffemodel -gpu 0 -method 0
  4. finetune quantized model:      ./decent_q finetune -model fix_results/float_train_test.prototxt -weights fix_results/float_train_test.caffemodel -gpu 0
  5. deploy finetuned fixed model:  ./decent_q depoly -model fix_train_test.prototxt -weights fix_finetuned.caffemodel -gpu 0


  Flags from tools/decent_q.cpp:
    -auto_test (Optional: test after calibration, need test dataset) type: bool
      default: false
    -calib_iter (Optional: max iterations for fix calibration) type: int32
      default: 100
    -classNUM (The number of segmentation classes.) type: int32 default: 19
    -data_bit (Optional: data bit width after fix) type: int32 default: 8
    -gpu (Optional: the GPU device id for calibration and test) type: string
      default: "0"
    -ignore_layers (Optinal: list of layers to be ignore during fix,
      comma-delimited) type: string default: ""
    -ignore_layers_file (Optional: YAML file which defines the layers to be
      ignore during fix, start with 'ignore_layers:') type: string default: ""
    -include_ip (Whether the InnerProduct layer is quantized.) type: bool
      default: false
    -input_blob (Optional: name of input data blob) type: string
      default: "data"
    -keep_bn (Optional: remain BatchNorm layers) type: bool default: true
    -method (Optional: method for fix, 0: OVERFLOW, 1: DIFF_S) type: int32
      default: 1
    -model (The model definition protocol buffer text file.) type: string
      default: ""
    -output_dir (Optional: Output directory for the fix results) type: string
      default: "./fix_results"
    -quan_bits (The number of bits used in quantization.) type: int32
      default: 8
    -segmentation (Optional; Segmentation testing using metric:classIOU)
      type: string default: ""
    -sigmoided_layers (Optinal: list of layers before sigmoid operation, to be
      fixed with optimization for sigmoid accuracy) type: string default: ""
    -solver (The solver for finetuning or training.) type: string default: ""
    -ssd (Optional: SSD testing with three AP computing styles: 11point,
      MaxIntegral, Integral) type: string default: ""
    -test_iter (Optional: max iterations for test) type: int32 default: 50
    -weights (The pretrained float weights for fix.) type: string default: ""
    -weights_bit (Optional: weights and bias bit width after fix) type: int32
      default: 8
    -yolo (Optional: YOLO testing with three AP computing styles: 11point,
      MaxIntegral, Integral) type: string default: ""
```
Fisihed the Host installation, Next we will steup target baord environment. Please unzip **2018-10-11-ZCU102-desktop-stretch.img.zip** to get the image file **2018-10-11-ZCU102-desktop-stretch.img**. Insert an empty SD-Card(with size >= 16 GB) into your local Ubuntu Linux Machine.

Execute below command

    sudo lsblk

on the Linux PC and search for your ZCU102 SD-Card(in some Ubuntu PCs the sd-card is seen as **/dev/ssd** in others as **/dev/mmcblk0**); let us assume it is called **/dev/sdd**

Burn the img file on the SD-Card with the Linux command:

    sudo dd if=deephi_zc102_dnndk_1.07.img of=/dev/ssd

(in case of troubles look at https://learn.sparkfun.com/tutorials/sd-cards-and-writing-images)
Normally it takes about 10-30 min depending on your PC, Once the above process has finished, you can boot the Target board

Open terminal(if requested, `login/password:root/root`) with the parameters(115200,8,n,1,N) for the UART connection. Connect a Ethernet cable form Target board to the local host Linux Machine, then run the following commannds:

    #On Target
    ifconfig eth0 192.168.1.100 netmask 255.255.255.0
In local Linux Machine:

    #On Local Host
    sudo ifconfig eth0 192.168.1.101 netmask 255.255.255.0

Copy realted packages from local machine to target board with the following commands:

    #On Local Host
    tar -xzvf deephi_dnndk_v2.08_beta.tar.gz
    cd deephi_dnndk_v2.08_beta
    tar -czvf common.tar.gz ./common
    tar -czvf ZCU102.tar.gz ./ZCU102
    scp common.tar.gz root@192.168.1.100:/root
    scp ZCU102.tar.gz root@192.168.1.100:/root
Move to terminal of Target board with following command:

    #On Target
    tar -xzvf deephi_dnndk_v2.08_beta.tar.gz
    cd /root
    tar -xzvf common.tar.gz
    tar -xzvf ZCU102.tar.gz
    cd ZCU102
    ./install.sh

Once finished you should test it on target baord with the following commands:

    dexplorer -v

You should see something similar:
 ``` console
DNNDK version 2.07 beta
Copyright @ 2016-2018 Deephi Inc. All Rights Reserved.

DExplorer version 1.5
Build Label: Ocy 12 2018 12:00:01

DSight version 1.4
Build Label: Oct 12 2018 12:00:02

N2Cube Core Library version 2.1
Build Label: Oct 12 2018 12:00:22

DPU Driver version 2.0
Build Label: Oct 12 2018 11:59:58
```
# Build and Delopy Network Model(Resnet50)
There are 5 developmet steps with DNNDK
-	Compression(Pruning and Quantization)
-	Compilation
-	DPU Application Development(Programming)
-	CPU+DPU Hybrid Compilation
-	Execution on evaluation board(Running)

**NOTES:DNNDK Pubilc Release version only support Quantization, there is no Pruning.**

Before running this example, please make sure you are on GCP DNNDK environment then download Resnet50 model file with the following commands:

    # On GCP
    wget http://www.deephi.com/assets/ResNet50.tar.gz
    tar zxvf ResNet50.tar.gz
    cd resnet50

### 1. Compression
Revised the decent_reset50.sh as following:

    #!/usr/bin/env bash
    
    #working directory
    work_dir=$(pwd)
    #path of float model
    model_dir=${work_dir}
    #output directory
    output_dir=${work_dir}/decent_output
    
    decent     quantize                               \
               -model ${model_dir}/float.prototxt     \
               -weights ${model_dir}/float.caffemodel \
               -output_dir ${output_dir} \
               -method 1

**NOTES: change the "fix" to "quantize"**

Then, you shoud be able to run decent_q to quantize the network.

    ./decent_reset50.sh

### 2. Compilation
Revised the dnnc_reset50.sh as following:

    #!/bin/bash
    net=resnet50
    model_dir=decent_output
    output_dir=dnnc_output
    
    echo "Compiling network: ${net}"
    
    dnnc --prototxt=${model_dir}/deploy.prototxt     \
           --caffemodel=${model_dir}/deploy.caffemodel \
           --output_dir=${output_dir}                  \
           --net_name=${net}                           \
           --dpu=4096FA                                \
           --cpu_arch=arm64


**NOTES: change dpu from 1152F to 4096FA and cpu_arch from arm32 to arm64**

Then, compile the network with following command:

    ./dnnc_reset50.sh

You should be able to see the output elf file in dnnc_output directory.

    ls -al dnnc_output/
    total 25896
    drwxr-xr-x 2 alex alex     4096 Dec 12 08:18 .
    drwxr-xr-x 5 alex alex     4096 Dec 12 08:17 ..
    -rw-rw-r-- 1 alex alex 24434400 Dec 12 08:18 dpu_resnet50_0.elf
    -rw-rw-r-- 1 alex alex  2068656 Dec 12 08:18 dpu_resnet50_2.elf

### 3. DPU Application Development
Develop source code to driver the running of neural network on CPU+DPU heterogeneous system with DNNDK lightweight C/C++ APIs. For Reset50 image classification, about total 200 lines C/C++ source code and only 50 lines related to DPU programming. In the example. the source locate on **/root/deephi_dnndk_v2.08_beta/ZCU102/sample/resnet50/src/main.cc** like as:
```console
int main(void) {
  /* DPU Kernels/Tasks for running ResNet50 */
  DPUKernel *kernelConv;
  DPUKernel *kernelFC;
  DPUTask *taskConv;
  DPUTask *taskFC;

  /* Attach to DPU driver and prepare for running */
  dpuOpen();
  /* Create DPU Kernels for CONV & FC Nodes in ResNet50 */
  kernelConv = dpuLoadKernel(KRENEL_CONV);
  kernelFC = dpuLoadKernel(KERNEL_FC);
  /* Create DPU Tasks for CONV & FC Nodes in ResNet50 */
  taskConv = dpuCreateTask(kernelConv, 0);
  taskFC = dpuCreateTask(kernelFC, 0);

  /* Run CONV & FC Kernels for ResNet50 */
  runResnet50(taskConv, taskFC);

  /* Destroy DPU Tasks & free resources */
  dpuDestroyTask(taskConv);
  dpuDestroyTask(taskFC);
  /* Destroy DPU Kernels & free resources */
  dpuDestroyKernel(kernelConv);
  dpuDestroyKernel(kernelFC);
  /* Dettach from DPU driver & free resources */
  dpuClose();

  return 0;
}
```
### 4. CPU+DPU Hybrid Compilation
Start the hybrid compilation process to compile & link code running on CPU and code running on DPU(generated by DNNC) then produce a final hybrid ELF executable binary. Before the hybrid compilation, you need to download the DPU elf file from GCP instance to local machine then transfer it to target board with following command:

    #On Local Host
    #transfer elf from GCP to local machine
    gcloud compute scp instance-1:/home/alex/ML/DNNDK/samples/resnet50/dnnc_output/dpu_resnet50_0.elf .
    gcloud compute scp instance-1:/home/alex/ML/DNNDK/samples/resnet50/dnnc_output/dpu_resnet50_2.elf .
    #transfer elf from local machine to target board
    scp dpu_resnet50_0.elf root@192.168.1.100:/root/ZCU102/samples/resnet50/model
    scp dpu_resnet50_2.elf root@192.168.1.100:/root/ZCU102/samples/resnet50/model

NOTES: Make sure the local host is connected to target board through Ethernet cable and network has been configured coorectly.

When these elf has been done to transfer then excute the hybrid compilation on target board.Before the hybrid compilation, please make sure the board have connected USB hub and then a mouse and keyboard.another, connect Ethernet cable to local host and DP cable to a monitor.Please use the keyboard and mouse of target board to open a terminal and input the following command:

    #On Target
    cd /root/deephi_dnndk_v2.08_beta/ZCU102/samples/resnet50
    make
    
### 5. Execution on evaluation board
In ZCU102 ubuntu desktop terminal with the following commands:

    #On Target
    ./reset50

You should be able to see the demo now.

# Quickly Installation by scripts
Finally, I crated above install step with the script file, let you can reduce time to install these toolchain
*	install_cuda90.sh --- This scripts are from CuDA to NCCL installation 
*	install_ristretto_caffe.sh --- This scripts are from Create Python2.7 Virtual Env. with ML-related package to Caffe-Ristretto installation

Now, excute the following Linux commands to install CuDA, CUDNN, NCCL and Ristretto:

    #On GCP
    cd ~
    sudo git clone https://github.com/xelalin/DNNDK_Installation
    source ~/DNNDK_Installation/scripts/caffe/install_cuda90.sh
    source ~/DNNDK_Installation/scripts/caffe/install_ristretto_caffe.sh

**When finished, please follow install DNNDK guide to compete all of installation.**


