#!/bin/bash


CKPTNAME=model.ckpt-50000
MOD_CLASS=hand

# Exit script on error.
set -e
# Echo each command, easier for debugging.
set -x

export PYTHONPATH=`pwd`:`pwd`/slim:$PYTHONPATH

# mobilenet v2
#python models/research/object_detection/export_inference_graph.py \
#--input_type image_tensor --pipeline_config_path TRAIN/learn/ckpt_custom/pipeline.config \
#--trained_checkpoint_prefix TRAIN/learn/train_custom/${CKPTNAME} --output_directory DETECTION_FROZEN_GRAPH

# faster rcnn resnet101
python models/research/object_detection/export_inference_graph.py \
--input_type image_tensor --pipeline_config_path TRAIN/learn/faster_rcnn_resnet101_coco_2018_01_28/pipeline_${MOD_CLASS}.config \
--trained_checkpoint_prefix TRAIN/learn/train_resnet_${MOD_CLASS}/${CKPTNAME} --output_directory DETECTION_FROZEN_GRAPH

# copy frozen graph in DETECTION_FROZEN_GRAPH_CLASS
#cp DETECTION_FROZEN_GRAPH/frozen_inference_graph.pb DETECTION_FROZEN_GRAPH_CLASS/frozen_inference_graph_${MOD_CLASS}.pb
