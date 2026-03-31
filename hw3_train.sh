#!/bin/bash


if [ "$1" = "not_adaption_val_org" ]
then
	python3 train.py --not_adaption --num_epochs $2 --data_root_dir $3 --val_dir org/val
elif [ "$1" = "not_adaption_val_fog" ]
then
	python3 train.py --not_adaption --num_epochs $2 --data_root_dir $3 --val_dir fog/val
else
	python3 train.py --num_epochs $2 --data_root_dir $3 --val_dir fog/val
fi
