#!/bin/bash

for file in *.xml; do
    new_folder="C\:\\\\lib\\\\tensorflow\\\\done_vio\\\\"
    #new_folder="I\:\\\\_selenium\\\\tst_deep_learning\\\\bicycle_rear_suspension\\\\"
    new_folder="E\:\\\\common_public\\\\projects\\\\deep_learning\\\\all\\\\"
    end_path_var=`cat "$file" | grep "<path>" | awk -F "\\\\" '{print $NF}'`
    sed -i "s~<path>.*$end_path_var~<path>"$new_folder$end_path_var"~g" "$file"
done
