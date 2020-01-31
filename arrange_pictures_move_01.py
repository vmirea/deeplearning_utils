"""proc_image.py

This script is used detect and create voc xml

larry's stuff
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import cv2
import tensorflow as tf
import glob
import ntpath
import os
import os.path
from os import path

#from xml.dom.minidom import parse, parseString
from lxml import etree
import xml.etree.ElementTree as ET

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

GENERATE_XML=True
DRAW_OBJECT=False
MOVE_FILE=False
OUTPUT_FOLFER = "results"
#I:\_selenium\donedeal\bicycles_folder
XML_FOLDES_OUTPUT = os.path.normpath(os.path.join(OUTPUT_FOLFER, "xml_out"))
JPG_FOLDES_OUTPUT = os.path.normpath(os.path.join(OUTPUT_FOLFER, "jpg_out"))
JPG_FOLDES_OUTPUT = os.path.normpath(os.path.join(OUTPUT_FOLFER, "I:\\_selenium\\donedeal\\road_bicycle_handlebar"))
JPG_CLASS_NOT_FOUND_FOLDES_OUTPUT = os.path.normpath(os.path.join(OUTPUT_FOLFER, "I:\\_selenium\\donedeal\\not_road_bicycle_handlebar"))
##I:\_selenium\donedeal\bycicles_folder

MOVED_CLAS_NAME = "road bicycles handlebar"

#PATH_TO_FROZEN_GRAPH = 'DETECTION_FROZEN_GRAPH/frozen_inference_graph_hand.pb'
#PATH_TO_LABELS = 'workspace/training_demo/labels_custom_detect.pbtxt'
#OUTPUT_PATH = 'detection_output.jpg'

#CLASS_OBJ = 'hand'
PATH_TO_FROZEN_GRAPH = 'FROZEN_GRAPS/frozen_inference_graph_vio.pb'
##PATH_TO_LABELS = 'workspace/training_demo/labels_custom_detect.pbtxt'
PATH_TO_LABELS = 'FROZEN_GRAPS\labels_class_vio.pbtxt'


import ntpath
import os
print(os.sep)
print(PATH_TO_FROZEN_GRAPH)
os.path.normpath(os.path.join(PATH_TO_FROZEN_GRAPH))
os.path.normpath(os.path.join(PATH_TO_LABELS))
print(PATH_TO_FROZEN_GRAPH)
print(os.sep)

#s.replace(os.sep,ntpath.sep)

# comment to enable back cuda/gpu
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

def detect_image(image_path,max_thresh,new_class):

#    PATH_TO_FROZEN_GRAPH = 'DETECTION_FROZEN_GRAPH_CLASS/frozen_inference_graph_' + my_class_name + '.pb'
#    PATH_TO_LABELS = 'DETECTION_FROZEN_GRAPH_CLASS/labels_class_' + my_class_name + '.pbtxt'

#    myfiles = glob.glob("images_test/*")
    myfiles = glob.glob(image_path + "**/*.jpg", recursive=True)
    print('myfiles = %s\n' % myfiles)
    
    #for my_file in myfiles:
    #    print(my_file)
    #    output_xmlfile_name = os.path.splitext(os.path.basename(my_file.replace('\\','/')))[0] + ".xml"
        #output_xmlfile_namepath = 'workspace/training_demo/annotations/' + output_xmlfile_name
        #print('XML FILE=' + output_xmlfile_namepath)
        #if path.isfile( output_xmlfile_namepath) :
        #    print("WARNING ::file annotation exist=%s" % output_xmlfile_namepath)
#            return
    
    # load label map
    category_index = label_map_util.create_category_index_from_labelmap(
        PATH_TO_LABELS)

    # load detection graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # define input/output tensors
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    print('Run inference..\n')
    cnt = 0
#    max_thresh = 0.60
        
    
    # run inference
    with detection_graph.as_default():
        with tf.Session() as sess:
            for my_file in myfiles:
                print('DETECT FILE number %d   :: %s' % (cnt, my_file))

                # :: XML ANNOTATION PASCAL VOC
                _myfile = my_file.replace('\\','/')
                myimgfile = ntpath.basename(_myfile)
                os.path.normpath(myimgfile)
                myimgfolder = os.path.basename(os.path.dirname(_myfile))
                print('myimgfolder = %s\n' % myimgfolder)
                output_xmlfile_name = os.path.splitext(os.path.basename(_myfile))[0] + ".xml"
                #os.path.normpath(output_xmlfile_name)
                print('output_xmlfile_name = %s\n' % output_xmlfile_name)
                #output_xmlfile_namepath = 'workspace/training_demo/annotations/' + output_xmlfile_name
                mynewxmlfile = os.path.normpath(os.path.join(XML_FOLDES_OUTPUT, output_xmlfile_name)) 
                os.path.normpath(mynewxmlfile)
                print('mynewxmlfile = %s\n' % mynewxmlfile)
                
                # this falg is to be used to append to voc xml instead of overwrite (if exist)
                is_xmlfile = False
                #text_path = 'E:/common_public/projects/deep_learning/network_test'
                text_path = 'workspace/text_path'
                os.path.normpath(text_path)
                #print('text_path = %s\n' % text_path)
                if path.isfile(mynewxmlfile) :
                    is_xmlfile = True
                    doc = etree.parse(mynewxmlfile)
                    root = doc.getroot()                    
                else:
                # create xml structure if no file
                #if is_xmlfile == False :                
                    # create XML 
                    root = etree.Element('annotation')
                    #root.append(etree.Element('child'))
                    
                    # :: XML :: FOLDER
                    child = etree.Element('folder')
                    child.text = myimgfolder    
                    root.append(child)
                    
                    # :: XML :: filename
                    child = etree.Element('filename')
                    child.text = myimgfile    
                    root.append(child)

                    # :: XML :: path
                    child = etree.Element('path')
                    child.text = text_path + _myfile    
                    root.append(child)

                    # source
                    child_source = etree.Element('source')
                    # :: XML :: image width
                    child = etree.Element('database')
                    child.text = 'Unknown'   
                    child_source.append(child)
                    root.append(child_source)

                # load input image
                img = cv2.imread(my_file)
                if img is None:
                    sys.exit('failed to load image: %s' % my_file)
                img = img[..., ::-1]  # BGR to RGB
                img_height, img_width, img_channels = img.shape

                if is_xmlfile == False :                
                    # image size
                    child_size = etree.Element('size')
                    # :: XML :: image width
                    child = etree.Element('width')
                    child.text = str(img_width)    
                    child_size.append(child)
                    # :: XML :: image height
                    child = etree.Element('height')
                    child.text = str(img_height)      
                    child_size.append(child)
                    # :: XML :: image depth
                    child = etree.Element('depth')
                    child.text = str(img_channels)      
                    child_size.append(child)
                    # now add size
                    root.append(child_size)

                    # :: XML :: segmented
                    child = etree.Element('segmented')
                    child.text = "0"    
                    root.append(child)


                ###############
                # RUN DETECTION
#                (boxes, scores, classes, num) = sess.run(
#                    [detection_boxes, detection_scores, detection_classes, num_detections],
#                    feed_dict={image_tensor: image_np_expanded})
          
                boxes, scores, classes, _ = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: np.expand_dims(img, 0)})
                #for class_name, score in zip(classes, scores):
                #    print(class_name, ':', score)

                # draw the results of the detection
                vis_util.visualize_boxes_and_labels_on_image_array(
                    img,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=6,
                    min_score_thresh=max_thresh)

                # This is the way I'm getting my coordinates
                boxes = np.squeeze(boxes)
                max_boxes_to_draw = boxes.shape[0]
                scores = np.squeeze(scores)
                min_score_thresh=max_thresh
                found_bicycle = False
                if GENERATE_XML:
                    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
                        class_id = np.squeeze(classes).astype(np.int32)[i]
                        # process only the elements from class MOVED_CLAS_NAME
                        if category_index[class_id]['name'] != MOVED_CLAS_NAME :
                            # shutil.copy(my_file, JPG_CLASS_NOT_FOUND_FOLDES_OUTPUT)
                            continue
                        #found_bicycle = True
                        if scores is None or scores[i] > min_score_thresh:
                            found_bicycle = True
                            # boxes[i] is the box which will be drawn
                            #print ("This box is gonna get used", boxes[i])
                            # xml :: object
                            child_obj = etree.Element('object')
                            # :: XML :: obj name
                            child = etree.Element('name')
                            #child.text = category_index[class_id]['name']
                            child.text = new_class
                            child_obj.append(child)
                            # :: XML :: obj pose
                            child = etree.Element('pose')
                            child.text = "Unspecified"
                            child_obj.append(child)
                            # :: XML :: obj truncated
                            child = etree.Element('truncated')
                            child.text = "0"
                            child_obj.append(child)
                            # :: XML :: obj difficult
                            child = etree.Element('difficult')
                            child.text = "0"
                            child_obj.append(child)
                            # :: XML :: obj bounding box
                            child_box = etree.Element('bndbox')
                            # :: XML :: obj bounding box :: xmin
                            child = etree.Element('xmin')
                            child.text = str(int(round(boxes[i, 1] * img_width)))
                            child_box.append(child)
                            # :: XML :: obj bounding box :: ymin
                            child = etree.Element('ymin')
                            child.text = str(int(round(boxes[i, 0] * img_height)))
                            child_box.append(child)
                            # :: XML :: obj bounding box :: xmax
                            child = etree.Element('xmax')
                            child.text = str(int(round(boxes[i, 3] * img_width)))
                            child_box.append(child)
                            # :: XML :: obj bounding box :: ymax
                            child = etree.Element('ymax')
                            child.text = str(int(round(boxes[i, 2] * img_height)))
                            child_box.append(child)
                            # append bounding box to object
                            child_obj.append(child_box)
                            # append object to main
                            root.append(child_obj)

                if GENERATE_XML:
                    if found_bicycle == True :
                        max_score = "{:.2f}".format(scores[0])
                        max_class_id = np.squeeze(classes).astype(np.int32)[0]
                        max_class_label = "NODET"
                        if scores[0] > max_thresh:
                            max_class_label = category_index[max_class_id]['name']
                        print('Max class Name %s\n' % max_class_label)
                        #current_file = img_paths[0]
                        shutil.move(my_file, JPG_FOLDES_OUTPUT)

                        # save the output image
                        img = img[..., ::-1]  # RGB to BGR
                        #myfilename = os.path.splitext(ntpath.basename(my_file))[0]
                        #output_jpgfile_name = os.path.normpath(os.path.join(JPG_FOLDES_OUTPUT, myfilename + '_' +  max_class_label + '_' + str(max_score) + '_' + str(cnt) + '.jpg'))
                        #cv2.imwrite(output_jpgfile_name, img)

                        #print('Output has been written to %s\n' % output_jpgfile_name)

                        # pretty string
                        s = etree.tostring(root, pretty_print=True)
                       # print(s) #generated xml

                        et = etree.ElementTree(root)
                        #et.write(sys.stdout, pretty_print=True)
                        et.write(mynewxmlfile, pretty_print=True)

                if Path(my_file).is_file():
                    shutil.move(my_file, os.path.join(JPG_CLASS_NOT_FOUND_FOLDES_OUTPUT, os.path.split(my_file)[-1]))
                cnt = cnt + 1

                
                # debug return after first n itterations
                #if cnt > 2000 :
                #    return


def main():
    image_path = "images_test"
    max_thresh = 0.99    
    new_class = "bicycle"
#    if len(sys.argv) != 2:
#        sys.exit('Usage: %s <image_path>' % sys.argv[0])
    if len(sys.argv) >= 3:
        new_class=sys.argv[1]
        image_path=sys.argv[2]
        image_path="I:\\_selenium\\donedeal\\bicycles_folder"
    else:
        sys.exit('Usage: %s <new_class> <image_path>' % sys.argv[0])
        
#    detect_image(image_path=sys.argv[1])
    detect_image(image_path,max_thresh,new_class)

if __name__ == '__main__':
    main()
