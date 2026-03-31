#!/bin/sh

python3 train.py --not_train --checkpoint_index $3 --infer_root_dir $1 --output_path $2
