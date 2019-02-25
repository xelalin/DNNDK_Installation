# set manually these 2 variables

export CAFFE_ROOT=$HOME/caffe_tools/Ristretto
export MYVENV=caffe_py27

#########################################################################
# install all the packges needed by Caffe itself
# (only once: if you install a second fork of caffe you do not need to install it twice)
#########################################################################
#deactivate
source ~/DNNDK_Installation/scripts/caffe/install_caffe_py27_packages.sh

#########################################################################
#create python2.7 virtual environment
#########################################################################
cd ~
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo python3 get-pip.py
sudo pip2 install virtualenv virtualenvwrapper
sudo rm -rf ~/.cache/pip get-pip.py

########################################################################
#Add the following lines to the ~/.bashrc file
########################################################################
echo '# Virtualenv and virtualenvwrapper' >> ~/.bashrc
echo 'export WORKON_HOME=$HOME/.virtualenvs' >> ~/.bashrc
echo 'export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python2' >> ~/.bashrc
echo 'export VIRTUALENVWRAPPER_VIRTUALENV=/usr/local/bin/virtualenv' >> ~/.bashrc
echo 'source /usr/local/bin/virtualenvwrapper.sh' >> ~/.bashrc
echo 'export PYTHONPATH=/usr/local/lib/python2.7/dist-packages:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc

mkvirtualenv $MYVENV -p python2
workon $MYVENV

#########################################################################
#########################################################################
# leave ALL the remaining lines below unchanged!
#########################################################################
#########################################################################

# install all the packages needed in the virtual env, which are compatible also with Caffe
sudo pip2 install -r ~/DNNDK_Installation/scripts/caffe/caffe_py27_requirements.txt

# patch for opencv 2.x
cd ~/.virtualenvs/$MYVENV/local/lib/python2.7/site-packages/
sudo ln -s /usr/lib/python2.7/dist-packages/cv2.x86_64-linux-gnu.so cv2.so
cd ~

#########################################################################
# clone Ristretto caffe from github
#########################################################################
cd ~
git clone https://github.com/pmgysel/caffe
mkdir ~/caffe_tools
mv ~/caffe ~/caffe_tools/Ristretto
cd ~

#########################################################################
# now compile caffe
#########################################################################
#workon $MYVENV
cd $CAFFE_ROOT
#cp Makefile orig_makefile
#cp Makefile.config orig_Makefile.config
cp ~/DNNDK_Installation/scripts/caffe/Makefile .
cp ~/DNNDK_Installation/scripts/caffe/Makefile.config .
cp ~/DNNDK_Installation/scripts/caffe/CMakeLists.txt .
#cp ~/scripts/caffe/test_layer_factory.cpp  ./src/caffe/test/test_layer_factory.cpp

export LC_ALL="C"

make clean
make all  -j8
make test -j8
make pycaffe
make distribute
make runtest -j8

########################################################################
#Add the following lines to the ~/.bashrc file
########################################################################
echo '# MACHINE LEARNING' >> ~/.bashrc
echo 'export CAFFE_ROOT=$HOME/caffe_tools/Ristretto' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CAFFE_ROOT/distribute/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PYTHONPATH=$CAFFE_ROOT/distribute/python:$PYTHONPATH' >> ~/.bashrc

source ~/.bashrc

#########################################################################
# now test everything
#########################################################################
workon $MYVENV

echo $CAFFE_ROOT
echo $LD_LIBRARY_PATH
echo $PYTHONPATH
python ~/DNNDK_Installation/scripts/caffe/test_caffe.py




