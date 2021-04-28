import os 
import time
import numpy as np
import cv2
import streamlit as st

from PIL import Image
from database.yolo.model.yolo_model import YOLO



def load_image(image_file):
    img=Image.open(image_file)
    return img


def process_image(img):
    ''' 이미지 리사이즈하고, 차원 화장
    img: 원본 이미지
    결과는 (64,64,3)으로 프로세싱한 이미지 반환'''
    image_org = cv2.resize(img, (416,416), interpolation = cv2.INTER_CUBIC)
    image_org = np.array(image_org, dtype='float32')
    image_org = image_org / 255.0
    image_org = np.expand_dims(image_org, axis = 0 )

    return image_org

def get_classess(file):
  '''   클래스의 이름을 리스트로 가져온다.   '''
  with open(file) as f :
    name_of_class = f.readlines()

  name_of_class = [  class_name.strip() for class_name in name_of_class  ]

  return name_of_class

def box_draw(image, boxes, scores, classes, all_classes):
    ''' 
    image : 오리지날 이미지 
    boxes : object의 박스 데이터, ndarray
    classes : object의 클래스 정보, ndarray
    scores : object의 확률, ndarray
    all_classes : 모든 클래스이 이름
    drow_box 를 그리는 함수'''
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print("------------")

def detect_image( image, yolo, all_classes):
    ''' image : 오리지날 이미지
    yolo : 욜로 모델
    all_classes : 전체 클라스 이름
    전체 이미지 리턴'''

    pimage =  process_image(image)

    image_boxes, image_classes, image_scores = yolo.predict(pimage, image.shape)

    if image_boxes is not None:
        box_draw(image, image_boxes, image_scores, image_classes, all_classes)
    
    return image




def yolo_detection():
    st.title('tesorflow image object detetion')

    st.subheader('이미지파일 업로드')
    image_file = st.file_uploader('Upload Image', #파일업로드 
                                type= ['png', 'jpg','jpeg']) #업로드 될 수 있는 이미지 파일
    if image_file is not None :
        img = load_image(image_file)
        st.image(img, width= 700)
        if st.button('Detection'):
            all_classes = get_classess('database/yolo/data/coco_classes.txt')
            yolo = YOLO(0.6,0.6)
            image = np.array(img)
            result_image= detect_image(image, yolo, all_classes)
            st.image(result_image, width= 700)
  