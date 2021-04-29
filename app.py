import streamlit as st
from Resource.Intro import Home
from Resource.ssd import ssd_inference
from Resource.segmentation import segmentic_inference
from Resource.yolo import yolo_inference

def main():

    st.title('Object Detection')

    menu = ['Home','SSD','YOLO','segmentation']

    choice = st.sidebar.selectbox('MENU',menu)

    if choice == 'Home':
        Home()

    elif choice == 'SSD':
        ssd_inference()
        # image_detection()
        # video_detection()
    
    elif choice == 'segmentation':
        segmentic_inference()
        # segementic_detection()

    elif choice == 'YOLO':
        yolo_inference()
        # result_image, class_dic= yolo_detection()
        # select_object(result_image, class_dic)

        

    else: 
        st.subheader('Introduce project')



if __name__ == '__main__' :
    main()