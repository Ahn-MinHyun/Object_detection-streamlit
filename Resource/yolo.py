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
    class_dic ={}
    i = 0
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
        class_dic[i] = [all_classes[cl], score, box]
        i += 1

    print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
    print(class_dic)
    print('box coordinate x,y,w,h: {0}'.format(box))

    print("------------")
    return class_dic

def detect_image( image, yolo, all_classes):
    ''' image : 오리지날 이미지
    yolo : 욜로 모델
    all_classes : 전체 클라스 이름
    전체 이미지 리턴'''

    pimage =  process_image(image)

    image_boxes, image_classes, image_scores = yolo.predict(pimage, image.shape)



    if image_boxes is not None:
         class_dic = box_draw(image, image_boxes, image_scores, image_classes, all_classes)
    # print('class:',image_classes)
    # print('image_scores',image_scores)
    
    return image, class_dic




def yolo_detection():
    st.title(' image object detetion')



    st.subheader('이미지파일 업로드')
    image_file = st.file_uploader('Upload Image', #파일업로드 
                                type= ['png', 'jpg','jpeg']) #업로드 될 수 있는 이미지 파일
    if image_file is not None :
        img = load_image(image_file)
        # st.image(img, width= 700)

        all_classes = get_classess('database/yolo/data/coco_classes.txt')
        st.sidebar.text('''object\n물체의 정확도에 대한 기준''')
        object_threshold = st.sidebar.slider('obj_threshold',min_value = 0.2, max_value=1.0, step = 0.1)
        st.sidebar.text('''non-maximum-suppression (IOU)\n같은 물체의 겹쳐지는 박스를 제거 하기 위한 기준''')
        nms_threshold = st.sidebar.slider('nms_threshold',min_value = 0.2, max_value=1.0, step = 0.1)
        yolo = YOLO(object_threshold,nms_threshold)

        st.sidebar.text("object_thresthod : "+ str(round(object_threshold,1)))
        st.sidebar.text("nms_thresthod : "+ str(round(nms_threshold,1)))

        image = np.array(img)
        result_image,  class_dic = detect_image(image, yolo, all_classes)
        

        select_list = {i[0] for i in class_dic.values()}
        select_class= st.sidebar.selectbox('데이터 표시', list(select_list))

        print(class_dic)
        st.sidebar.subheader('물체 정확도')
        for i, (class_name, class_score,box)  in class_dic.items():
            st.sidebar.write('**'+class_name+'**'+ '  '+ str(round(class_score,2)))



        for i in class_dic.values():
            if i[0] in select_class :
                box = i[2]
                x, y, w, h = box
                top = max(0, np.floor(x + 0.5).astype(int))
                left = max(0, np.floor(y + 0.5).astype(int))
                right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
                bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

                cv2.rectangle(result_image, (top, left), (right, bottom), (135, 250, 51), 2)

        st.image(result_image, width= 700)


