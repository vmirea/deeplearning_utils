#!/bin/sh
image_prefix=$1
mkdir output||true
for f in *.jpeg; do mv -- "$f" "${f%.\*}.jpg"; done
for f in *.JPG; do mv -- "$f" "${f%.\*}.jpg"; done

#tmp_stamp=$(date +%T);ls *.jpg | sed -e "s/\..*\./_$(date +%s)_/g"
old_path="I:\\\\_selenium\\\\done\\\\bmx\\\\"
new_path=$old_path"output\\\\"
for file in *.jpg; do
    tmp_stamp=$(date +%m%d%s%3N)
    no_ext_name=`echo $file | awk -F.jpg '{print $1}'`
    new_name=`echo $file | sed -e 's/\..*\./_'$tmp_stamp'\./g'`
    new_name=`echo $new_name | awk -F.jpg '{print $1}'`
    new_name=$image_prefix$new_name 
    #mv "$file" "$(basename "$file" .html).txt"
    echo "no_ext_name="$no_ext_name
    echo "new_name="$new_name
    #cp "$no_ext_name".* output
    #mv 'output/'$no_ext_name'.xml' 'output/'$new_name'.xml'
    #mv 'output/'$no_ext_name'.jpg' 'output/'$new_name'.jpg'
    mv $no_ext_name'.jpg' 'output/'$new_name'.jpg'
    #sed -i 's/'$no_ext_name'/'$new_name'/' 'output/'$new_name'.xml'
    #sed -i "s~$no_ext_name~$new_name~" 'output/'$new_name'.xml'
    #sed -i "s~$old_path~$new_path~" "output/"$new_name'.xml'
done
#ls *.jpg | sed -e "s/\..*\./_$tmp_stamp\./g"
#tmp_stamp=$(date +%T);ls *.jpg | sed -e "s/\..*\./_$(date +%s)_/g"
