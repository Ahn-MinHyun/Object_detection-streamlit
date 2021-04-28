import streamlit as st
from PIL import Image


def load_image(image_file):
    img=Image.open(image_file)
    return img


def Home():
    st.title('Open CV, 인공지능 활용한 Object detection')
    st.write('인공지능을 이용해 카메라로 들어오는 파일들의 물체를 구분/확인한다.')
    result_img=load_image('database/image/result.png')
    st.image(result_img)

