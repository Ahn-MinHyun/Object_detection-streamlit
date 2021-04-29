import streamlit as st
from Resource.Intro import Home
from Resource.ssd import image_detection, video_detection
from Resource.segmentation import  segementic_detection
from Resource.yolo import yolo_detection, select_object

def main():

    st.title('Object Detection')

    menu = ['Home','SSD','YOLO','segmentation']

    choice = st.sidebar.selectbox('MENU',menu)

    if choice == 'Home':
        Home()

    elif choice == 'SSD':
        image_detection()
        video_detection()
    
    elif choice == 'segmentation':
        segementic_detection()

    elif choice == 'YOLO':
        result_image, class_dic= yolo_detection()
        select_object(result_image, class_dic)

        

    else: 
        st.subheader('Introduce project')



if __name__ == '__main__' :
    main()