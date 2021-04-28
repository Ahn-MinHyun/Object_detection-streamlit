import numpy as np
import argparse
import imutils
import time
import cv2
import os
import matplotlib.pyplot as plt
import streamlit as st
import streamlit as st
from PIL import Image



SET_WIDTH = int(600)
normalize_image = 1 / 255.0
resize_image_shape = (1024, 512)


def load_image(image_file):
    img=Image.open(image_file)
    return img


def segementic_detection():
    st.title('Image Segmentic segmentation')

    
    
    st.subheader('분할하고 싶은 이미지')
    st.write('**TensorFlow로 Cityscapes 데이터세트에서 학습된 ENet을 활용하였습니다.**')
    image_file = st. file_uploader('Upload Image', #파일업로드 
    type= ['png', 'jpg','jpeg']) #업로드 될 수 있는 이미지 파일
    if image_file is not None :
        st.write(image_file.name)
        img = load_image(image_file)
        st.image(img, width= 700)

        sample_img = np.array(img)
        # print(sample_img.shape)
        sample_img = imutils.resize(sample_img, width=SET_WIDTH)
        # opencv 의 pre trained model 을 통해서, 예측하기 위해서는
        # (opencv의 DNN라이브러리를 이용하기 위해서)
        # 입력이미지를 blob 으로 바꿔줘야 한다.
        blob_img = cv2.dnn.blobFromImage(sample_img, normalize_image, 
                                        resize_image_shape, 0, 
                                        swapRB = True, crop=False)
        # Enet 모델 가져오기.
        cv_enet_model = cv2.dnn.readNet('database/enet-cityscapes/enet-model.net')
        # print( cv_enet_model)    

        # blob된 이미지를 모델에 맞게 세팅해주는 함수 
        cv_enet_model.setInput(blob_img)

        # 클라스의 갯수 많큼 아웃풋이 나온다. 
        cv_enet_model_output = cv_enet_model.forward()

        # # 1 : 1개의 이미지를 넣었으므로
        # # 20 : 클래스의 갯수
        # # 512 : 행렬의 행의 갯수
        # # 1024 : 행렬의 열의 갯수
        # print(cv_enet_model_output.shape)

        # 레이블 이름을 로딩
        label_values = open('database/enet-cityscapes/enet-classes.txt').read().split('\n')
        # print(label_values)
        # 더미 데이타 제거
        label_values = label_values[ : -2+1]

        # 원래의 모양인 (1, 20, 512, 1024) 에 있는 값을, 변수로 저장.
        # 20은 클래스의 갯수, 512는 높이 1024는 너비로 저장.
        IMG_OUTPUT_SHAPE_START = 1 
        IMG_OUTPUT_SHAPE_END = 4
        classes_num, h, w = cv_enet_model_output.shape[IMG_OUTPUT_SHAPE_START : IMG_OUTPUT_SHAPE_END]

        # 중요 2 모델의 아웃풋 20개 행렬을 하나의 행렬로 만든다.(1장에서 비교할 곳은 (20,514,1024)중 20이니깐 
        class_map = np.argmax(cv_enet_model_output[0], axis = 0)

        # 색정보를 로딩
        CV_ENET_SHAPE_IMG_COLORS = open('database/enet-cityscapes/enet-colors.txt').read().split('\n')
        # 더미 데이타 제거
        CV_ENET_SHAPE_IMG_COLORS = CV_ENET_SHAPE_IMG_COLORS[ : -2+1]
        CV_ENET_SHAPE_IMG_COLORS = np.array([np.array(color.split(',')).astype('int')  for color in CV_ENET_SHAPE_IMG_COLORS  ])
        # print(CV_ENET_SHAPE_IMG_COLORS)

        ## 중요 3 하나의 행렬을 => 이미지로 만든다.
        # 각 픽셀별로, 클래스에 해당하는 숫자가 적힌 class_map을
        # 각 숫자에 매핑되는 색깔로 셋팅해 준것이다.
        # 따라서 각 픽셀별 색깔 정보가 들어가게 되었다.
        # 2차원 행렬을, 3차원 채널이 있는 RGB 행렬로 만든다.
        mask_class_map = CV_ENET_SHAPE_IMG_COLORS[class_map]

        # 리사이즈 한다.
        # 인터폴레이션을 INTER_NEAREST 로 한 이유는?? 
        # 레이블 정보(0~19) 와 컬러정보 (23,100,243) 는 둘다 int 이므로, 
        # 가장 가까운 픽셀 정보와 동일하게 셋팅해주기 위해서.
        mask_class_map = cv2.resize(mask_class_map, (sample_img.shape[1], sample_img.shape[0]) , 
                interpolation = cv2.INTER_NEAREST )
        class_map = cv2.resize(class_map, (sample_img.shape[1], sample_img.shape[0]) , 
                            interpolation=cv2.INTER_NEAREST)


        # 원본이미지랑, 색마스크 이미지를 합쳐서 보여준다.
        # 가중치 비율을 줘서 보여준다. 
        mask_slide = st.sidebar.slider('mask',min_value=0.3, max_value=1.0, step=0.05)
        # print(mask_slide)
        cv_enet_model_output = ( ( (1-mask_slide) * sample_img ) + ( mask_slide * mask_class_map) ).astype('uint8')
        

        # 라벨 가져오기
        my_legend = np.full(( len(label_values) * 25 ,  300 , 3 ) , 255  , dtype='uint8' )
        # my_legend = np.zeros( ( len(label_values) * 25 ,  300 , 3 )   , dtype='uint8' )
        
        for ( i, (class_name, img_color)) in enumerate( zip(label_values , CV_ENET_SHAPE_IMG_COLORS)) :
            color_info = [  int(color) for color in img_color  ] 
            cv2.putText(my_legend, class_name, (6, (i*25) + 16) , 
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0) , 1 )
            cv2.rectangle(my_legend, (200, (i*25)), (300, (i*25) + 20) , tuple(color_info),-1)

        # print("my_lengend",my_legend)
        # # cv2.imshow('color',mask_class_map)

        st.sidebar.image(my_legend,width=300, caption = 'color bar')
        st.image(cv_enet_model_output,width=700)





