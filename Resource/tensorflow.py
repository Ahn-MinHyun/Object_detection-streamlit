import streamlit as st
import cv2
import os
import pathlib
import numpy as np

import tempfile
import six.moves.urllib as urllib
import sys
# import tarfilest
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# 버전 호환성을 위한 코드
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

# 모델 가져오는 함수 
def load_model():
    model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

    model_dir = pathlib.Path(model_dir)/"saved_model"

    model = tf.saved_model.load(str(model_dir))

    return model



# 모델에 맞게 이미지를 맞춰주는 함수
def run_inference_for_single_image(model, image):
    # 넘파이 어레이로 바꿈
    # image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = tf.convert_to_tensor(output_dict['detection_masks'], dtype=tf.float32)
        output_dict['detection_boxes'] = tf.convert_to_tensor(output_dict['detection_boxes'], dtype=tf.float32)
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1],resize_method='nearest')  
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                        tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        
    return output_dict

## 함수 테스트
# image_np = cv2.imread('data/images/image1.jpg')
# output_dict = run_inference_for_single_image(detection_model, image_np)
# print(output_dict)

# 이미지를 보여주는 함수 
def show_inference(model, image_np):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    # image_np = cv2.imread(str(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    PATH_TO_LABELS = 'database/models/research/object_detection/data/mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.array(output_dict['detection_boxes']),
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed',None),
        use_normalized_coordinates=True,
        line_thickness=8)
    print('what : ',output_dict['detection_classes'])
    print('what1 : ',output_dict['detection_scores'])
    print('what2 :', output_dict)
    st.image(image_np, width= 600)
    # cv2.imshow("result",image_np)

def load_image(image_file):
    img=Image.open(image_file)
    return img


def image_detection():
    ## 라벨 불러오기
    # List of the strings that is used to add correct label for each box.
    # print('라벨')

    # 모델 불러오기
    detection_model = load_model()

    st.title('tesorflow image object detetion')

    st.subheader('이미지파일 업로드')
    image_file = st. file_uploader('Upload Image', #파일업로드 
                type= ['png', 'jpg','jpeg']) #업로드 될 수 있는 이미지 파일
    if image_file is not None :
        st.write(image_file.name)
        img = load_image(image_file)
        pix = np.array(img)
        show_inference(detection_model, pix)
        # print(pix)

def video_detection():
    pass
#     st.title('tesorflow video object detetion')

#     st.subheader('비디오파일 업로드')
#     video_file =  st.file_uploader('Upload Video', #파일업로드 
#     type= ['mp4','avi'])
#     if video_file is not None :
#         video_bytes = video_file.read()
#         st.video(video_bytes)
#         if st.button('Detection',key='video'):
#             tfile = tempfile.NamedTemporaryFile(delete=False) 
#             tfile.write(video_file.read())
#             cap = cv2.VideoCapture(tfile.name)

#             if cap.isOpened() == False :
#                 print( "Error opening video stream or file" )

#             else :
#                 while cap.isOpened() :
#                     # 사진을 1장씩 가져와서.
#                     ret, frame = cap.read()

#                     # 제대로 사진 가져왔으면, 화면에 표시
#                     if ret == True :

#                         show_inference(detection_model, frame)          
                        
#                     else :
#                         print("Can't receive frame (stream end?). Exiting ...")
#                         break