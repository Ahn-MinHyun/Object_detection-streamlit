import streamlit as st
from Resource.Intro import Home
from Resource.ssd import ssd_inference, image_detection
from Resource.segmentation import segmentic_inference, segementic_detection
from Resource.yolo import yolo_inference, yolo_detection, select_object

def main():

    st.title('Object Detection')

    menu = ['Home','SSD','YOLO','segmentation']

    choice = st.sidebar.selectbox('MENU',menu)

    if choice == 'Home':
        Home()

    elif choice == 'SSD':
        ssd_inference()
        image_detection()

    
    elif choice == 'segmentation':
        segmentic_inference()
        segementic_detection()

    elif choice == 'YOLO':
        yolo_inference()
        result_image, class_dic= yolo_detection()
        select_object(result_image, class_dic)

        

    else: 
        st.subheader('Introduce project')



if __name__ == '__main__' :
    main()