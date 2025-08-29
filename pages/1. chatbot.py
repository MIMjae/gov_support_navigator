import streamlit as st
from model_kosbert import generate_answer as generate_recommendation

## 2025.07 버전
# st.set_page_config(page_title="🤖 AI Chatbot (Demo)", layout="wide")
# st.title("🤖 맞춤형 정부 지원사업 안내 챗봇")

# # 세션 상태 초기화
# if "message_list" not in st.session_state:
#     st.session_state.message_list = []

# # 이전 메시지 출력
# for message in st.session_state.message_list:
#     with st.chat_message(message["role"]):  # user 또는 ai
#         st.write(message["content"])

# # 입력 처리
# if user_input := st.chat_input("내용을 입력하여 대화를 시작하세요."):
#     # 사용자 메시지
#     with st.chat_message("user"): # , avatar="👤"
#         st.write(user_input)
#     st.session_state.message_list.append({"role": "user", "content": user_input})

#     # 모델 응답
#     with st.spinner("답변을 생성 중입니다."):
#         response = generate_answer(user_input)
#         with st.chat_message("ai"): # , avatar="🤖"
#             st.write(response)
#         st.session_state.message_list.append({"role": "ai", "content": response})



## 2025.08 버전
# --- 페이지 설정 ---
st.set_page_config(page_title="🤖 AI 챗봇", layout="wide")
st.title("🤖 맞춤형 정부 지원사업 안내 챗봇")

# --- 사이드바 ---
with st.sidebar:
    st.header("🤖 AI 모델 선택")
    selected_model = st.selectbox(
        "원하는 작업 유형을 선택하세요.",
        ("사업 공고 추천 (Ko-SBERT)", "일반 대화 (준비 중)") # 다른 모델 추가 가능
    )

# --- AI 응답 생성 로직 ---
def get_ai_response(user_input, model_type):
    if model_type == "사업 공고 추천 (Ko-SBERT)":
        return generate_recommendation(user_input)
    elif model_type == "일반 대화 (준비 중)":
        # return generate_general_answer(user_input)
        return "일반 대화 모델은 현재 준비 중입니다."
    else:
        return "모델을 선택해주세요."

# --- 메인 챗봇 UI ---

# 세션 상태 초기화
if "message_list" not in st.session_state:
    st.session_state.message_list = []

# 이전 메시지 출력
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) # st.write 대신 st.markdown 사용

# 입력 처리
if user_input := st.chat_input("기업의 특징이나 필요한 지원 내용을 알려주세요."):
    # 사용자 메시지
    st.session_state.message_list.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # 모델 응답
    with st.spinner("답변을 생성 중입니다..."):
        response = get_ai_response(user_input, selected_model)
        st.session_state.message_list.append({"role": "ai", "content": response})
        with st.chat_message("ai"):
            st.markdown(response) # st.write 대신 st.markdown 사용