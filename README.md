# farm-tools

Programs, library functions, tools working on top of deepset-ai's FARM

See https://github.com/deepset-ai/FARM

My own code for using the FARM library, opinionated, some parts may be out of date.

Contains modified modules from the FARM package to make some of the things I needed work.

License of this code is the same as for the FARM software.

## Installation/Setup

See [`conda-create-env-sh`](conda-create-env.sh)

NOTE: make sure to either run the `conda-create-env.sh` code or perform the following steps as close as possible 
to get repeatable results!

* create environment, tested with python 3.8
* activate environment
* Instead of directly installing FARM we need to slightly modify the installation process by changing the 
  requirements file. This is done like this:
  * from within the farm-tools root dir, clone the FARM repo: `git clone https://github.com/deepset-ai/FARM.git`
  * copy our modified farm-requirements.txt into FARM/requirements.txt : `cp farm-requirements.txt FARM/requirements.txt`
  * do a development install directly from the cloned repo:
  * `cd FARM`
  * `pip install -r requirements.txt`
  * `pip install -e .`
  * `cd ..`
* if necessary because the wrong version of pytorch got installed for your system, 
  uninstall pytorch installed as part of FARM and reinstall the version that you / that fits your configuration
* Install the additional dependencies needed for farm-tools:
  * `pip install -r farm-tool-requirements.txt`
* Install the farm-tool package:
  * `pip install -e .`
  * if this is done the commands can be executed as e.g. `farm-estimate` instead 
    of `$PATH_TO_FARMTOOLS/farm_tools/farm_estimate.py` 
* Install the jupyter kernel:
  `python -m ipykernel install --user --name=farm-tools`


## Usage

All commands provide usage information with the parameter `--help`

#### Perfomance estimation: `farm-estimate`

```
usage: farm-estimate [-h] --runname RUNNAME --infile INFILE [--cfg CFG] [--seed SEED] [--n_gpu N_GPU] [--use_cuda USE_CUDA] [--use_amp USE_AMP]
                     [--do_lower_case DO_LOWER_CASE] [--text_column TEXT_COLUMN] [--batch_size BATCH_SIZE] [--max_seq MAX_SEQ]
                     [--deterministic DETERMINISTIC] [-d] [--label_column LABEL_COLUMN] [--dev_splt DEV_SPLT] [--grad_acc GRAD_ACC] [--lm_dir LM_DIR]
                     [--lm_name LM_NAME] [--evaluate_every EVALUATE_EVERY] [--max_epochs MAX_EPOCHS] [--dropout DROPOUT] [--lrate LRATE]
                     [--es_patience ES_PATIENCE] [--es_metric ES_METRIC] [--es_mode ES_MODE] [--es_min_evals ES_MIN_EVALS] [--es_hd ES_HD]
                     [--labels LABELS] [--dev_stratification DEV_STRATIFICATION] [--fts FTS] [--fts_cfg [FTS_CFG [FTS_CFG ...]]] [--fos FOS]
                     [--fos_cfg [FOS_CFG [FOS_CFG ...]]] [--hd_dim HD_DIM] [--hd0_cfg [HD0_CFG [HD0_CFG ...]]] [--hd1_cfg [HD1_CFG [HD1_CFG ...]]]
                     [--hd2_cfg [HD2_CFG [HD2_CFG ...]]] [--hd3_cfg [HD3_CFG [HD3_CFG ...]]] [--hd4_cfg [HD4_CFG [HD4_CFG ...]]]
                     [--losses_alpha LOSSES_ALPHA] [--eval_method EVAL_METHOD] [--xval_folds XVAL_FOLDS] [--holdout_repeats HOLDOUT_REPEATS]
                     [--holdout_train HOLDOUT_TRAIN] [--eval_stratification EVAL_STRATIFICATION]

optional arguments:
  -h, --help            show this help message and exit
  --runname RUNNAME     Experiment name. Files are stored in directory {runname}-{datetime}
  --infile INFILE       Path to input file
  --cfg CFG             Path to configuration file
  --seed SEED           Random seed (42)
  --n_gpu N_GPU         Number of GPUs, if GPU is to be used (1
  --use_cuda USE_CUDA   If GPUs should be used, if not specified, determined from setup
  --use_amp USE_AMP     Use AMP (False
  --do_lower_case DO_LOWER_CASE
                        Lower case tokens (False)
  --text_column TEXT_COLUMN
                        Name of in/out text column (text)
  --batch_size BATCH_SIZE
                        Batch size (32)
  --max_seq MAX_SEQ     Maximum sequence length (whatever the trainer used)
  --deterministic DETERMINISTIC
                        Use deterministic (slower) code (False)
  -d                    Enable debug mode
  --label_column LABEL_COLUMN
                        Name of label column (target)
  --dev_splt DEV_SPLT   Development set proportion (0.1)
  --grad_acc GRAD_ACC   Gradient accumulation steps (1)
  --lm_dir LM_DIR       Load LM from that directory instead of default
  --lm_name LM_NAME     Load LM from that known named model (will download and cache model!)
  --evaluate_every EVALUATE_EVERY
                        Evaluate every this many batches (10)
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs (20)
  --dropout DROPOUT     Dropout rate (0.2)
  --lrate LRATE         Learning rate (5e-06)
  --es_patience ES_PATIENCE
                        Early stopping patience (10)
  --es_metric ES_METRIC
                        Early stopping metric (f1_micro)
  --es_mode ES_MODE     Early stopping mode (max)
  --es_min_evals ES_MIN_EVALS
                        Early stopping minimum evaluation steps (1)
  --es_hd ES_HD         Early stopping head number to use (0)
  --labels LABELS       Comma separated list of labels, if missing, assume '0' and '1'
  --dev_stratification DEV_STRATIFICATION
                        Use stratified dev set splits? (False)
  --fts FTS             FarmTasks class to use (FTSingleClassification)
  --fts_cfg [FTS_CFG [FTS_CFG ...]]
                        FarmTasks configuration settings of the form parm=value
  --fos FOS             FarmOptSched class to use (FOSDefault)
  --fos_cfg [FOS_CFG [FOS_CFG ...]]
                        Farm optimizer/scheduler configuration settings of the form parm=value
  --hd_dim HD_DIM       Dimension of the LM output, i.e. the head input (768)
  --hd0_cfg [HD0_CFG [HD0_CFG ...]]
                        Head 0 config parameters of the form parm=value
  --hd1_cfg [HD1_CFG [HD1_CFG ...]]
                        Head 1 config parameters of the form parm=value
  --hd2_cfg [HD2_CFG [HD2_CFG ...]]
                        Head 2 config parameters of the form parm=value
  --hd3_cfg [HD3_CFG [HD3_CFG ...]]
                        Head 2 config parameters of the form parm=value
  --hd4_cfg [HD4_CFG [HD4_CFG ...]]
                        Head 2 config parameters of the form parm=value
  --losses_alpha LOSSES_ALPHA
                        Alpha for loss aggregation (weight of head 0, weight for head 1 is 1-alpha)
  --eval_method EVAL_METHOD
                        Evaluation method, one of xval, holdout (xval)
  --xval_folds XVAL_FOLDS
                        Number of folds for xval (10)
  --holdout_repeats HOLDOUT_REPEATS
                        Number of repetitions for holdout estimation (5)
  --holdout_train HOLDOUT_TRAIN
                        Portion used for training for holdout estimation (0.7)
  --eval_stratification EVAL_STRATIFICATION
                        Use stratified samples for the evaluation splits? (False)
```

#### Hyperparameter Search: `farm-hsearch`

```
usage: farm-hsearch [-h] --runname RUNNAME --infile INFILE [--cfg CFG] [--seed SEED] [--n_gpu N_GPU] [--use_cuda USE_CUDA] [--use_amp USE_AMP]
                    [--do_lower_case DO_LOWER_CASE] [--text_column TEXT_COLUMN] [--batch_size BATCH_SIZE] [--max_seq MAX_SEQ]
                    [--deterministic DETERMINISTIC] [-d] [--label_column LABEL_COLUMN] [--dev_splt DEV_SPLT] [--grad_acc GRAD_ACC] [--lm_dir LM_DIR]
                    [--lm_name LM_NAME] [--evaluate_every EVALUATE_EVERY] [--max_epochs MAX_EPOCHS] [--dropout DROPOUT] [--lrate LRATE]
                    [--es_patience ES_PATIENCE] [--es_metric ES_METRIC] [--es_mode ES_MODE] [--es_min_evals ES_MIN_EVALS] [--es_hd ES_HD]
                    [--labels LABELS] [--dev_stratification DEV_STRATIFICATION] [--fts FTS] [--fts_cfg [FTS_CFG [FTS_CFG ...]]] [--fos FOS]
                    [--fos_cfg [FOS_CFG [FOS_CFG ...]]] [--hd_dim HD_DIM] [--hd0_cfg [HD0_CFG [HD0_CFG ...]]] [--hd1_cfg [HD1_CFG [HD1_CFG ...]]]
                    [--hd2_cfg [HD2_CFG [HD2_CFG ...]]] [--hd3_cfg [HD3_CFG [HD3_CFG ...]]] [--hd4_cfg [HD4_CFG [HD4_CFG ...]]]
                    [--losses_alpha LOSSES_ALPHA] [--eval_method EVAL_METHOD] [--xval_folds XVAL_FOLDS] [--holdout_repeats HOLDOUT_REPEATS]
                    [--holdout_train HOLDOUT_TRAIN] [--eval_stratification EVAL_STRATIFICATION] --hcfg HCFG --outpref OUTPREF [--halg HALG]
                    [--halg_rand_n HALG_RAND_N] [--beamsize BEAMSIZE] [--est_var EST_VAR] [--est_cmp EST_CMP]

optional arguments:
  -h, --help            show this help message and exit
  --runname RUNNAME     Experiment name. Files are stored in directory {runname}-{datetime}
  --infile INFILE       Path to input file
  --cfg CFG             Path to configuration file
  --seed SEED           Random seed (42)
  --n_gpu N_GPU         Number of GPUs, if GPU is to be used (1
  --use_cuda USE_CUDA   If GPUs should be used, if not specified, determined from setup
  --use_amp USE_AMP     Use AMP (False
  --do_lower_case DO_LOWER_CASE
                        Lower case tokens (False)
  --text_column TEXT_COLUMN
                        Name of in/out text column (text)
  --batch_size BATCH_SIZE
                        Batch size (32)
  --max_seq MAX_SEQ     Maximum sequence length (whatever the trainer used)
  --deterministic DETERMINISTIC
                        Use deterministic (slower) code (False)
  -d                    Enable debug mode
  --label_column LABEL_COLUMN
                        Name of label column (target)
  --dev_splt DEV_SPLT   Development set proportion (0.1)
  --grad_acc GRAD_ACC   Gradient accumulation steps (1)
  --lm_dir LM_DIR       Load LM from that directory instead of default
  --lm_name LM_NAME     Load LM from that known named model (will download and cache model!)
  --evaluate_every EVALUATE_EVERY
                        Evaluate every this many batches (10)
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs (20)
  --dropout DROPOUT     Dropout rate (0.2)
  --lrate LRATE         Learning rate (5e-06)
  --es_patience ES_PATIENCE
                        Early stopping patience (10)
  --es_metric ES_METRIC
                        Early stopping metric (f1_micro)
  --es_mode ES_MODE     Early stopping mode (max)
  --es_min_evals ES_MIN_EVALS
                        Early stopping minimum evaluation steps (1)
  --es_hd ES_HD         Early stopping head number to use (0)
  --labels LABELS       Comma separated list of labels, if missing, assume '0' and '1'
  --dev_stratification DEV_STRATIFICATION
                        Use stratified dev set splits? (False)
  --fts FTS             FarmTasks class to use (FTSingleClassification)
  --fts_cfg [FTS_CFG [FTS_CFG ...]]
                        FarmTasks configuration settings of the form parm=value
  --fos FOS             FarmOptSched class to use (FOSDefault)
  --fos_cfg [FOS_CFG [FOS_CFG ...]]
                        Farm optimizer/scheduler configuration settings of the form parm=value
  --hd_dim HD_DIM       Dimension of the LM output, i.e. the head input (768)
  --hd0_cfg [HD0_CFG [HD0_CFG ...]]
                        Head 0 config parameters of the form parm=value
  --hd1_cfg [HD1_CFG [HD1_CFG ...]]
                        Head 1 config parameters of the form parm=value
  --hd2_cfg [HD2_CFG [HD2_CFG ...]]
                        Head 2 config parameters of the form parm=value
  --hd3_cfg [HD3_CFG [HD3_CFG ...]]
                        Head 2 config parameters of the form parm=value
  --hd4_cfg [HD4_CFG [HD4_CFG ...]]
                        Head 2 config parameters of the form parm=value
  --losses_alpha LOSSES_ALPHA
                        Alpha for loss aggregation (weight of head 0, weight for head 1 is 1-alpha)
  --eval_method EVAL_METHOD
                        Evaluation method, one of xval, holdout (xval)
  --xval_folds XVAL_FOLDS
                        Number of folds for xval (10)
  --holdout_repeats HOLDOUT_REPEATS
                        Number of repetitions for holdout estimation (5)
  --holdout_train HOLDOUT_TRAIN
                        Portion used for training for holdout estimation (0.7)
  --eval_stratification EVAL_STRATIFICATION
                        Use stratified samples for the evaluation splits? (False)
  --hcfg HCFG           TOML configuration file for the hyperparameter search (required)
  --outpref OUTPREF     Output prefix for the files written for the hsearch run
  --halg HALG           Search algorithm, one of grid, random, greedy, beam (grid)
  --halg_rand_n HALG_RAND_N
                        Number of random runs for halg=random (20)
  --beamsize BEAMSIZE   Size of beam for halg=beam (3)
  --est_var EST_VAR     Estimation variable to use for sorting/searching (head0_f1_macro_mean)
  --est_cmp EST_CMP     Comparison to use for optimizing est_var, min or max (max)
```

#### Train a model: `farm-train`

```buildoutcfg
usage: farm-train [-h] --runname RUNNAME --infile INFILE [--cfg CFG] [--seed SEED] [--n_gpu N_GPU] [--use_cuda USE_CUDA] [--use_amp USE_AMP]
                  [--do_lower_case DO_LOWER_CASE] [--text_column TEXT_COLUMN] [--batch_size BATCH_SIZE] [--max_seq MAX_SEQ]
                  [--deterministic DETERMINISTIC] [-d] [--label_column LABEL_COLUMN] [--dev_splt DEV_SPLT] [--grad_acc GRAD_ACC] [--lm_dir LM_DIR]
                  [--lm_name LM_NAME] [--evaluate_every EVALUATE_EVERY] [--max_epochs MAX_EPOCHS] [--dropout DROPOUT] [--lrate LRATE]
                  [--es_patience ES_PATIENCE] [--es_metric ES_METRIC] [--es_mode ES_MODE] [--es_min_evals ES_MIN_EVALS] [--es_hd ES_HD]
                  [--labels LABELS] [--dev_stratification DEV_STRATIFICATION] [--fts FTS] [--fts_cfg [FTS_CFG [FTS_CFG ...]]] [--fos FOS]
                  [--fos_cfg [FOS_CFG [FOS_CFG ...]]] [--hd_dim HD_DIM] [--hd0_cfg [HD0_CFG [HD0_CFG ...]]] [--hd1_cfg [HD1_CFG [HD1_CFG ...]]]
                  [--hd2_cfg [HD2_CFG [HD2_CFG ...]]] [--hd3_cfg [HD3_CFG [HD3_CFG ...]]] [--hd4_cfg [HD4_CFG [HD4_CFG ...]]]
                  [--losses_alpha LOSSES_ALPHA]

optional arguments:
  -h, --help            show this help message and exit
  --runname RUNNAME     Experiment name. Files are stored in directory {runname}-{datetime}
  --infile INFILE       Path to input file
  --cfg CFG             Path to configuration file
  --seed SEED           Random seed (42)
  --n_gpu N_GPU         Number of GPUs, if GPU is to be used (1
  --use_cuda USE_CUDA   If GPUs should be used, if not specified, determined from setup
  --use_amp USE_AMP     Use AMP (False
  --do_lower_case DO_LOWER_CASE
                        Lower case tokens (False)
  --text_column TEXT_COLUMN
                        Name of in/out text column (text)
  --batch_size BATCH_SIZE
                        Batch size (32)
  --max_seq MAX_SEQ     Maximum sequence length (whatever the trainer used)
  --deterministic DETERMINISTIC
                        Use deterministic (slower) code (False)
  -d                    Enable debug mode
  --label_column LABEL_COLUMN
                        Name of label column (target)
  --dev_splt DEV_SPLT   Development set proportion (0.1)
  --grad_acc GRAD_ACC   Gradient accumulation steps (1)
  --lm_dir LM_DIR       Load LM from that directory instead of default
  --lm_name LM_NAME     Load LM from that known named model (will download and cache model!)
  --evaluate_every EVALUATE_EVERY
                        Evaluate every this many batches (10)
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs (20)
  --dropout DROPOUT     Dropout rate (0.2)
  --lrate LRATE         Learning rate (5e-06)
  --es_patience ES_PATIENCE
                        Early stopping patience (10)
  --es_metric ES_METRIC
                        Early stopping metric (f1_micro)
  --es_mode ES_MODE     Early stopping mode (max)
  --es_min_evals ES_MIN_EVALS
                        Early stopping minimum evaluation steps (1)
  --es_hd ES_HD         Early stopping head number to use (0)
  --labels LABELS       Comma separated list of labels, if missing, assume '0' and '1'
  --dev_stratification DEV_STRATIFICATION
                        Use stratified dev set splits? (False)
  --fts FTS             FarmTasks class to use (FTSingleClassification)
  --fts_cfg [FTS_CFG [FTS_CFG ...]]
                        FarmTasks configuration settings of the form parm=value
  --fos FOS             FarmOptSched class to use (FOSDefault)
  --fos_cfg [FOS_CFG [FOS_CFG ...]]
                        Farm optimizer/scheduler configuration settings of the form parm=value
  --hd_dim HD_DIM       Dimension of the LM output, i.e. the head input (768)
  --hd0_cfg [HD0_CFG [HD0_CFG ...]]
                        Head 0 config parameters of the form parm=value
  --hd1_cfg [HD1_CFG [HD1_CFG ...]]
                        Head 1 config parameters of the form parm=value
  --hd2_cfg [HD2_CFG [HD2_CFG ...]]
                        Head 2 config parameters of the form parm=value
  --hd3_cfg [HD3_CFG [HD3_CFG ...]]
                        Head 2 config parameters of the form parm=value
  --hd4_cfg [HD4_CFG [HD4_CFG ...]]
                        Head 2 config parameters of the form parm=value
  --losses_alpha LOSSES_ALPHA
                        Alpha for loss aggregation (weight of head 0, weight for head 1 is 1-alpha)
```

#### Apply a trained model: `farm-apply`

```
usage: farm-apply [-h] --infile INFILE [--cfg CFG] [--seed SEED] [--n_gpu N_GPU] [--use_cuda USE_CUDA] [--use_amp USE_AMP]
                  [--do_lower_case DO_LOWER_CASE] [--text_column TEXT_COLUMN] [--batch_size BATCH_SIZE] [--max_seq MAX_SEQ]
                  [--deterministic DETERMINISTIC] [-d] --outfile OUTFILE --modeldir MODELDIR [--label_column LABEL_COLUMN]
                  [--prob_column PROB_COLUMN] [--num_processes NUM_PROCESSES]

optional arguments:
  -h, --help            show this help message and exit
  --infile INFILE       Path to input file
  --cfg CFG             Path to configuration file
  --seed SEED           Random seed (42)
  --n_gpu N_GPU         Number of GPUs, if GPU is to be used (1
  --use_cuda USE_CUDA   If GPUs should be used, if not specified, determined from setup
  --use_amp USE_AMP     Use AMP (False
  --do_lower_case DO_LOWER_CASE
                        Lower case tokens (False)
  --text_column TEXT_COLUMN
                        Name of in/out text column (text)
  --batch_size BATCH_SIZE
                        Batch size (32)
  --max_seq MAX_SEQ     Maximum sequence length (whatever the trainer used)
  --deterministic DETERMINISTIC
                        Use deterministic (slower) code (False)
  -d                    Enable debug mode
  --outfile OUTFILE     Path to output TSV file
  --modeldir MODELDIR   Path to directory where the model is stored
  --label_column LABEL_COLUMN
                        Name of added label column (prediction)
  --prob_column PROB_COLUMN
                        Name of added probability column (prob)
  --num_processes NUM_PROCESSES
                        Number of processes to use (1)
```
