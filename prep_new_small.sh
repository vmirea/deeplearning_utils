#!/bin/bash

# Exit script on error.
set -e
# Echo each command, easier for debugging.
set -x


export PYTHONPATH=`pwd`:`pwd`/slim:$PYTHONPATH


source "$PWD/constants.sh"


echo "PREPARING label map..."

#python create_pascal_tf_record_custom.py 
python models/research/object_detection/dataset_tools/create_pascal_tf_record_custom.py \
--data_dir=workspace/training_demo/ \
--annotations_dir=anno_train/ \
--output_path=TRAIN/learn/data_custom/pas_train_small_hand2.record \
--label_map_path=workspace/training_demo/labels_custom_hand_train.pbtxt

