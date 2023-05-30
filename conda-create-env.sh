#!/bin/bash

envname=$1
if [[ "x$envname" == "x" ]]
then
  envname=farm-tools
fi
echo Trying to create environment $envname

set -e 
set -o pipefail
trap "echo Terminating script, something went wrong!!!" EXIT


d=`which conda || /bin/true` 
if [[ "x$d" == "x" ]]
then
    echo no conda !!!
    exit 1
fi
echo conda found at $d

d=`dirname $d`
c=$d/../etc/profile.d/conda.sh

if [ ! -f $c  ]
then
  echo not found $c
  echo run conda init bash
  exit 1
fi

. $c

conda create -y -n $envname python=3.8
conda init bash
conda activate $envname

echo Activated $envname running from `pwd`

# we cannot pip install FARM any longer since its requirements are broken
# instead we will clone farm, update the requirements with our own and then install locally

# are we in the correct directory?
# was farm-requirements.tmp
if [ ! -f farm-requirements.txt ]
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

python -m ipykernel install --user --name=$envname

trap - EXIT

echo conda environment $envname and ipykernel $envname created successfully
