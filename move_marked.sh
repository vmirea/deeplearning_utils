#!/bin/bash

mkdir marked_folder||true

for file in *.xml; do
    tmp_stamp=$(date +%Y%m%d%s%3N)
    no_ext_name=`echo $file | awk -F.xml '{print $1}'`
    if [ -e $no_ext_name.jpg ]
    then
        if [ -e $no_ext_name.xml ]
        then
            mv $no_ext_name'.jpg' marked_folder
            mv $no_ext_name'.xml' marked_folder
        fi
    fi
done