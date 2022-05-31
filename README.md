# farm-tools
Programs, library functions, tools working on top of deepset-ai's FARM

See https://github.com/deepset-ai/FARM


My own code for using the FARM library, opinionated, some parts may be out of date.

Contains modified modules from the FARM package to make some of the things I needed work.

License of this code is the same as for the FARM software.

## Installation/Setup

See [`conda.sourceme`](conda.sourceme)

* create environment
* install FARM
* if necessary, uninstall pytorch installed as part of FARM and reinstall the version that you / that fits your configuration
* install additional dependencies
* (optional) from this directory, install farm-tools itself: `pip install -e .`

## Usage

All commands provide usage information with the parameter `--help`

Performance estimation (xval, holdout):

Example:
```commandline
./farm_eval.py --runname expname --infile tsvfile.tsv --text_column colname 
   --batch_size 32 --max_seq 96 --label_column labelcol --lm_dir langmodeldir 
   --eval_method xval --xval_folds 5 --eval_stratification true 
```
