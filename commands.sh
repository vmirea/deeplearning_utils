python.exe E:\common_public\projects\deep_learning\detection_util_scripts\generate_csv.py xml E:\common_public\projects\deep_learning\bicycle_parts\train_network\common_road_bicycles_handlebars\annotations\xmls\ E:\common_public\projects\deep_learning\bicycle_parts\train_network\common_road_bicycles_handlebars\csv\output_single_file.csv
=> E:\common_public\projects\deep_learning\bicycle_parts\train_network\common_road_bicycles_handlebars\csv\output_single_file.csv
 python.exe E:\common_public\projects\deep_learning\detection_util_scripts\generate_pbtxt.py csv E:\common_public\projects\deep_learning\bicycle_parts\train_network\common_road_bicycles_handlebars\csv\output_single_file.csv E:\common_public\projects\deep_learning\bicycle_parts\train_network\common_road_bicycles_handlebars\annotations\label_map.pbtxt
=> E:\common_public\projects\deep_learning\bicycle_parts\train_network\common_road_bicycles_handlebars\label_map.pbtxt
 C:\cygwin64\bin\ls.exe  images/*.jpg | C:\cygwin64\bin\sed.exe -e 's/.jpg//g' | C:\cygwin64\bin\sed.exe -e "s/images\///g" >  E:\common_public\projects\deep_learning\bicycle_parts\train_network\common_road_bicycles_handlebars\annotations\trainval.txt
python.exe E:\common_public\projects\deep_learning\detection_util_scripts\generate_tfrecord.py E:\common_public\projects\deep_learning\bicycle_parts\train_network\common_road_bicycles_handlebars\csv\output_single_file.csv E:\common_public\projects\deep_learning\bicycle_parts\train_network\common_road_bicycles_handlebars\annotations\label_map.pbtxt E:\common_public\projects\deep_learning\bicycle_parts\train_network\common_road_bicycles_handlebars\images\  E:\common_public\projects\deep_learning\bicycle_parts\train_network\common_road_bicycles_handlebars\output.tfrecords
python E:\common_public\projects\deep_learning\detection_util_scripts\generate_yolo_txt.py E:\common_public\projects\deep_learning\bicycle_parts\train_network\common_road_bicycles_handlebars\csv\output_single_file.csv E:\common_public\projects\deep_learning\bicycle_parts\train_network\common_road_bicycles_handlebars\output\

python E:\common_public\projects\deep_learning\detection_util_scripts\generate_train_eval.py -o E:\common_public\projects\deep_learning\bicycle_parts\train_network\common_road_bicycles_handlebars\train_eval\ E:\common_public\projects\deep_learning\bicycle_parts\train_network\common_road_bicycles_handlebars\csv\output_single_file.csv
python.exe E:\common_public\projects\deep_learning\bicycle_parts\train_network\common_road_bicycles_handlebars\create_tf_record.py
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_inception_v2_coco.config

#E:\common_public\projects\deep_learning\bicycle_parts\train_network\forks_with_suspension> python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_inception_v2_coco.config   

#python.exe E:\common_public\projects\deep_learning\detection_util_scripts\generate_tfrecord.py E:\common_public\projects\deep_learning\bicycle_parts\train_network\common_road_bicycles_handlebars\csv\output_single_file.csv E:\common_public\projects\deep_learning\bicycle_parts\train_network\common_road_bicycles_handlebars\label_map.pbtxt E:\common_public\projects\deep_learning\bicycle_parts\train_network\common_road_bicycles_handlebars\images\  E:\common_public\projects\deep_learning\bicycle_parts\train_network\common_road_bicycles_handlebars\output.tfrecords
