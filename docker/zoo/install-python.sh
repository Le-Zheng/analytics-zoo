#!/usr/bin/env bash

set -e

PY_VERSION_SUFFIX=3
PYTHON=python${PY_VERSION_SUFFIX}
PIP=pip${PY_VERSION_SUFFIX}

apt-get install -y ${PYTHON}-minimal
apt-get install -y build-essential ${PYTHON} ${PYTHON}-setuptools ${PYTHON}-dev ${PYTHON}-pip

${PIP} install --upgrade setuptools
${PIP} install numpy scipy
${PIP} install --no-binary pandas -I pandas
${PIP} install scikit-learn matplotlib seaborn jupyter wordcloud moviepy requests h5py opencv-python tensorflow==1.15.0
${PIP} install torch==1.1.0 torchvision==0.3.0 -f https://download.pytorch.org/whl/torch_stable.html

ln -s /usr/bin/python3 /usr/bin/python
ln -s /usr/bin/pip3 /usr/local/bin/pip
#Fix tornado await process
pip uninstall -y -q tornado
pip install tornado==5.1.1

${PYTHON} -m ipykernel.kernelspec