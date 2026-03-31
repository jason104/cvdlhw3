command:

	bash hw3_train.sh $1 $2 $3

explanation:

	$1: training type, If domain adaptive training: "adaption"
			   elif Source training and validate on source: "not_adaption_val_org"
			   elif Source training and validate on target: "not_adaption_val_fog"
	$2: num_epochs, e.g. 9
	$3: data_root_dir, the folder path before "org" and "fog", e.g., "./hw3_dataset"

Example:
	bash hw3_train.sh adaption 9 ./hw3_dataset

