# model_kosbert.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pickle
import os

MODEL_NAME = 'jhgan/ko-sbert-nli'
EMBEDDING_PATH = './models/program_embeddings.npy'
DATA_PATH = './models/programs_data.pkl'
SIMILARITY_THRESHOLD = 0.7  # <-- 유사도 임계값 설정 (60% 이상)

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_saved_data():
    if os.path.exists(EMBEDDING_PATH) and os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'rb') as f:
            programs_data = pickle.load(f)
        program_embeddings = np.load(EMBEDDING_PATH)
        return programs_data, program_embeddings
    return None, None

def generate_answer(user_query: str) -> str:
    model = load_model()
    programs_data, program_embeddings = load_saved_data()

    if programs_data is None:
        return "죄송합니다. 추천 데이터 파일(`embeddings.npy`, `data.pkl`)을 찾을 수 없습니다."

    # --- 2. 유사도 계산 및 필터링 (핵심 변경 부분) ---
    query_embedding = model.encode([user_query])
    similarities = cosine_similarity(query_embedding, program_embeddings)[0]

    # (인덱스, 유사도) 쌍으로 묶은 후, 유사도 높은 순으로 정렬
    sorted_indices_with_scores = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    
    # 임계값(Threshold)을 넘는 모든 결과를 필터링
    high_score_results = []
    for index, score in sorted_indices_with_scores:
        if score >= SIMILARITY_THRESHOLD:
            high_score_results.append({
                "program": programs_data[index],
                "score": score
            })

    # --- 3. 결과 개수에 따른 동적 문구 생성 ---
    num_high_score_results = len(high_score_results)

    if num_high_score_results == 0:
        return f"죄송합니다. 관련성이 높은 사업 공고를 찾지 못했습니다."

    # 조건에 따라 안내 문구 변경
    if num_high_score_results >= 5:
        response_text = f"관련성 높은 사업은 {num_high_score_results}개 이며 그중 제일 관련성 높은 사업 5개에 대해 알려드립니다.\n\n"
        # 최종 결과는 최대 5개로 제한
        final_results = high_score_results[:5]
    else:
        response_text = f"요청하신 내용과 가장 관련성 높은 사업 공고 {num_high_score_results}개 이며 다음과 같습니다.\n\n"
        # 5개 미만이면 찾은 결과 모두 표시
        final_results = high_score_results

    # --- 4. 최종 결과 텍스트 생성 ---
    for i, item in enumerate(final_results, start=1):
        program = item["program"]
        score = item["score"]
        response_text += f"{i}. **{program.get('공고명', 'N/A')}**\n"
        response_text += f"   - **분야**: {program.get('지원분야', 'N/A')}\n"
        response_text += f"   - **대상**: {program.get('모집 대상', 'N/A')}\n"
        response_text += f"   - **자격**: {program.get('모집 자격', 'N/A')}\n"
        response_text += f"   - **링크**: [바로가기]({program.get('공고상세URL', '#')})\n\n"
            
    return response_text