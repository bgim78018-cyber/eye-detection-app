import streamlit as st
import tensorflow as tf
import urllib.request
import os
import cv2
import numpy as np

# 웹사이트 타이틀 및 설명문 꾸미기
st.set_page_config(page_title="졸음 감지 시스템", page_icon="👁️")
st.title("👁️ 실시간 눈 깜빡임 및 졸음 감지 시스템")
st.write("아래 웹캠 켜기 버튼을 누르고 카메라를 응시해 주세요.")

# 구글 드라이브에서 100MB 넘는 인공지능을 다운로드 받아 웹서버에 장착하는 최적화 함수
@st.cache_resource
def load_model():
    model_path = 'eye_model.h5'
    if not os.path.exists(model_path):
        with st.spinner("구글 드라이브 창고에서 대용량 인공지능 모델을 가져오는 중입니다... (최초 1회만 실행됩니다)"):
            # 구글 드라이브 ID 입력 부분
            file_id = '1O-N7Tu51f_YAzVFGvw7u3wuiUtaXBlYk'
            url = f"https://docs.google.com/uc?export=download&id={file_id}"
            urllib.request.urlretrieve(url, model_path)
    return tf.keras.models.load_model(model_path)

# 인공지능 로드
model = load_model()

# 스트림릿 내장 웹캠 캡처 가젯 활성화
img_file_buffer = st.camera_input("눈을 떴다 감았다 하며 테스트해보세요!")

if img_file_buffer is not None:
    # 촬영된 이미지를 인공지능이 읽을 수 있는 데이터로 변환
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # 코랩 학습 규격인 224x224 크기로 카메라 화면 실시간 리사이징
    resized_img = cv2.resize(cv2_img, (224, 224)) 
    normalized_img = resized_img / 255.0  # 정규화
    input_data = np.expand_dims(normalized_img, axis=0) # 4차원 배열로 확장
    
    # 인공지능 예측 수행
    with st.spinner("눈 상태 분석 중..."):
        prediction = model.predict(input_data)
    
    # 결과가 0.5보다 크면 눈을 뜬 상태, 낮으면 감은 상태로 판정 출력
    if prediction[0][0] > 0.5:
        st.success(f"🟢 안전 상태 : 눈을 올바르게 뜨고 있습니다! (확률: {prediction[0][0]*100:.1f}%)")
    else:
        st.error(f"🔴 졸음 경고 : 지금 눈을 감고 계십니다!! (확률: {(1-prediction[0][0])*100:.1f}%)")