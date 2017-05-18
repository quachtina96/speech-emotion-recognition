#!/bin/bash
# sets up environment

unset SSH_ASKPASS

echo "adding python modules"
module load engaging/python/3.5.0
pip3 install --user h5py
pip3 install --user pandas
pip3 install --user keras
pip3 install --user numpy
pip3 install --user pydot
pip3 install --user graphviz
pip3 install --user tensorflow-gpu
pip3 install --user theano
pip3 install --user --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp35-cp35m-linux_x86_64.whl
#apt-get install graphviz
echo "done setting up python"