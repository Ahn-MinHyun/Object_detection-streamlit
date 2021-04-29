import streamlit as st
from PIL import Image



def Home():
    st.subheader('''Open CV, Tensorflow를 활용한 Object detection''')
    

    st.write('인공지능을 이용해 카메라로 들어오는 파일들의 물체를 인식합니다.')

    result_img=Image.open('database/image/result.png')
    st.image(result_img)

    st.write('연속 처리하여 동영상의 물체들을 구분할 수 있습니다.')
    
    st.text('서버의 사양이 낮아 동영상 변환 불가로 동영상첨부')
    seg_video = open('database/video/segmantic.mp4','rb').read()
    st.video(seg_video)
    video_file = open('database/video/output.mp4','rb').read()
    st.video(video_file)

    st.write('''detection 모델들은 Bounding box를 만들고 각 box에 있는 feature를 extract한 후 classifier를 적용합니다. 
    하지만 이러한 과정은 *real-time으로 적용하기에 느리고, 임베디드화 시키기에도 연산량이 너무 많다*는 단점이 있습니다.''')
    


# ---------YOLO---------------------------
    st.title('YOLO')
    st.write("""YOLO란 우리가 흔히 You Only Look Once의 약자로 
    기존의 Object detection 속도가 real-time으로 사용하기에는 느리다는 문제점을 해결하기 위해 나온 알고리즘입니다. 
    YOLO의 가장 큰 특징은 Image를 bounding box를 찾을때와 classification을 
    따로하는 것이 아니라 **두가지를 한번에 한다**는 것입니다.""")
    yolo_reference = Image.open('database/image/yolo.png')
    st.image(yolo_reference)

# -----------SSD------------------------------
    st.title('SSD')
    st.write("""YOLO는 빠르다는 장점이 있지만 정확도가 떨어진다는 단점이 있다. 
    SSD는 이러한 단점까지 보안한 Model입니다.""")
    ssd_reference= Image.open('database/image/ssd.png')
    st.image(ssd_reference)
    st.write('SSD의 핵심은 다수의 conv feature map의 각 cell으로부터 category score와 box offset값을 예측하는 것입니다.')

# -------------SEGMENTIC-----------------------
    st.title('Segmentic segmentation')
    st.image('http://ataspinar.com/wp-content/uploads/2017/11/deeplearing_types.png')
    st.write('''Detection이 물체가 있는 위치를 찾아서 물체에 대해 Boxing을 하는 것이였다면, 
    Segmentation이란 한 그림 안에서 **영역**을 나누는 방식입니다.''')
    seg_reference = Image.open('database/image/Segementation.png')
    st.image(seg_reference)

    


