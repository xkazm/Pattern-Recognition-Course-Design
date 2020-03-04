import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tarfile
from tensorflow.models.research.object_detection.utils.label_map_util import load_labelmap,convert_label_map_to_categories,create_category_index
from tensorflow.models.research.object_detection.utils.ops import reframe_box_masks_to_image_masks
from tensorflow.models.research.object_detection.utils.visualization_utils import visualize_boxes_and_labels_on_image_array

MODEL_NAME='mask_rcnn_inception_v2_coco_2018_01_28'
MODEL_FILE=MODEL_NAME+'.tar'

PATH_TO_CKPT=MODEL_NAME+'/frozen_inference_graph.pb'

PATH_TO_LABELS=os.path.join('F:/seven/models/research/object_detection/data','mscoco_label_map.pbtxt')

NUM_CLASSES=90
tar_file=tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name=os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file,os.getcwd())

detection_graph=tf.Graph()
with detection_graph.as_default():
    od_graph_def=tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT,'rb') as fid:
        serialized_graph=fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def,name='')

label_map=load_labelmap(PATH_TO_LABELS)
categories=convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES,use_display_name=True)
category_index=create_category_index(categories)

def run_inference_for_single_image(image,graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops=tf.get_default_graph().get_operations()
            all_tensor_names={output.name for op in ops for output in op.outputs}
            tensor_dict={}
            for key in ['num_detections','detection_boxes','detection_scores','detection_classes','detection_masks']:
                tensor_name=key+':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key]=tf.get_default_graph().get_tensor_by_name(tensor_name)

            if 'detection_masks' in tensor_dict:
                detection_boxes=tf.squeeze(tensor_dict['detection_boxes'],[0])
                detection_masks=tf.squeeze(tensor_dict['detection_masks'],[0])

                real_num_detection=tf.cast(tensor_dict['num_detections'][0],tf.int32)
                detection_boxes=tf.slice(detection_boxes,[0,0],[real_num_detection,-1])
                detection_masks=tf.slice(detection_masks,[0,0,0],[real_num_detection,-1,-1])
                detection_masks_reframed=reframe_box_masks_to_image_masks(
                    detection_masks,detection_boxes,image.shape[0],image.shape[1]
                )
                detection_masks_reframed=tf.cast(
                    tf.greater(detection_masks_reframed,0.5),tf.uint8
                )
                tensor_dict['detection_masks']=tf.expand_dims(
                    detection_masks_reframed,0
                )
            image_tensor=tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            output_dict=sess.run(tensor_dict,feed_dict={image_tensor:np.expand_dims(image,0)})

            output_dict['num_detections']=int(output_dict['num_detections'][0])
            output_dict['detection_classes']=output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes']=output_dict['detection_boxes'][0]
            output_dict['detection_scores']=output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks']=output_dict['detection_masks'][0]
    return output_dict

if __name__=="__main__":
    image=cv2.imread('F:/seven/prDesign/fdata/daisy/image_0851.jpg')

    output_dict=run_inference_for_single_image(image,detection_graph)

    visualize_boxes_and_labels_on_image_array(
        image,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinated=True,
        line_thickness=8
    )