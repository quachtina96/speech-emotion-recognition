#!/bin/bash
# sets up environment

echo "adding python modules"

module load engaging/python/3.5.1
pip3 install --user pandas
pip3 install --user keras
pip3 install --user numpy
pip3 install --user pydot
pip3 install --user graphviz
#apt-get install graphviz

echo "done setting up python"
