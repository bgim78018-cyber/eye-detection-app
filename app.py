import streamlit as st
import onnxruntime as ort
import urllib.request
import os
import cv2
import numpy as np

st.set_page_config(page_title="졸음 감지 시스템", page_icon="👁️")
st.title("👁️ 초경량 실시간 눈 깜빡임 감지 시스템")

# 구글 드라이브에서 초경량 ONNX 모델 가져오기
@st.cache_resource
def load_onnx_model():
    model_path = 'eye_model.onnx'
    if not os.path.exists(model_path):
        with st.spinner("구글 드라이브에서 초경량 인공지능을 가져오는 중..."):
            file_id = '1L_-Q-JcYxwp5w4VVR6ubgiEjc--mmeDG'
            url = f"https://docs.google.com/uc?export=download&id={file_id}"
            urllib.request.urlretrieve(url, model_path)
    return ort.InferenceSession(model_path)

ort_session = load_onnx_model()

# 웹캠 UI
img_file_buffer = st.camera_input("눈을 떴다 감았다 하며 테스트해보세요!")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # 1. 크기 조절 및 색상 변환
    resized_img = cv2.resize(cv2_img, (224, 224))
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    
    # 2. 정규화 복구 (0~1 범위로 조정)
    normalized_img = resized_img.astype(np.float32) / 255.0
    
    # 3. 차원 순서는 텐서플로 형태(1, 224, 224, 3) 유지
    input_data = np.expand_dims(normalized_img, axis=0)
    
    # ONNX 인프런스 실행
    input_name = ort_session.get_inputs()[0].name
    prediction = ort_session.run(None, {input_name: input_data})
    
    # 결과 추출 및 출력
    result = prediction[0][0][0]
    
    # 디버깅용 실시간 생 출력값 확인
    st.write(f"🤖 모델의 내부 출력값(raw): {result:.4f}")
    
    if result > 0.5:
        st.success(f"🟢 안전 상태 : 눈을 뜨고 있습니다! (확률: {result*100:.1f}%)")
    else:
        st.error(f"🔴 졸음 경고 : 눈을 감고 계십니다!! (확률: {(1-result)*100:.1f}%)")
