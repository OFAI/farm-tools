#!/bin/bash

set -e 
set -o pipefail

d=`which conda`
if [[ "x$d" == "x" ]]
then
    echo no conda
    exit 1
fi

d=`dirname $d`
c=$d/../etc/profile.d/conda.sh

if [ ! -f $c  ]
then
  echo not found $c
  echo run conda init bash
  exit 1
fi

. $c

conda create -y -n farm-tools python=3.8
conda activate farm-tools

# we cannot pip install FARM any longer since its requirements are broken
# instead we will clone farm, update the requirements with our own and then install locally

# are we in the correct directory?
if [ ! -f farm-requirements.tmp ]
then
   echo Please run this from farm-tools root directory
   exit 1
fi

git clone https://github.com/deepset-ai/FARM.git
cp farm-requirements.txt FARM/requirements.txt
pushd FARM
pip install -r requirements.txt
pip install -e .

popd

pip install -r farm-tool-requirements.txt
pip install -e .

python -m ipykernel install --user --name=farm-tools

echo conda environment farm-tooks and ipykernel farm-tools created successfully
