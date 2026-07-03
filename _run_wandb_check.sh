#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dicode
export WANDB_API_KEY=
export WANDB_DISABLE_CODE=true
cd ~
python _check_restore_smooth.py
