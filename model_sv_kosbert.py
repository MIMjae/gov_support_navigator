# search_model_fast.py (기존 코드를 수정한 버전)

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

MODEL_NAME = 'jhgan/ko-sbert-nli'
EMBEDDING_PATH = 'program_embeddings.npy'
DATA_PATH = '/models/programs_data.pkl'

# --- 모델 및 미리 계산된 데이터 로드 ---
print("🔍 모델과 저장된 데이터를 로드합니다...")
model = SentenceTransformer(MODEL_NAME)

# 저장된 파일이 있는지 확인 후 로드
if os.path.exists(EMBEDDING_PATH) and os.path.exists(DATA_PATH):
    with open(DATA_PATH, 'rb') as f:
        programs_data = pickle.load(f)
    program_embeddings = np.load(EMBEDDING_PATH)
    print("✅ 모델과 데이터를 성공적으로 로드했습니다.")
else:
    print(f"🚨 오류: '{EMBEDDING_PATH}' 또는 '{DATA_PATH}' 파일을 찾을 수 없습니다.")
    print("먼저 create_embeddings.py 스크립트를 실행하여 파일을 생성해주세요.")
    # 파일이 없으면 실행을 멈추거나, 여기서 생성 로직을 다시 호출할 수 있습니다.
    exit()


def find_similar_programs(user_query: str, top_n: int = 3) -> str:
    """
    미리 로드된 임베딩을 사용해 유사 공고를 빠르게 검색합니다.
    """
    query_embedding = model.encode([user_query])
    similarities = cosine_similarity(query_embedding, program_embeddings)
    top_indices = np.argsort(similarities[0])[::-1][:top_n]
    results = [programs_data[i] for i in top_indices]

    response_text = f"'{user_query}'와 가장 관련성 높은 사업 공고는 다음과 같습니다.\n"
    # (출력 포맷팅 로직은 이전과 동일)
    for i, program in enumerate(results):
        response_text += f"\n{i+1}. {program.get('공고명', 'N/A')}\n"
        response_text += f"   - 분야: {program.get('지원분야', 'N/A')}\n"
        response_text += f"   - 대상: {program.get('모집 대상', 'N/A')}\n"
        response_text += f"   - 자격: {program.get('모집 자격', 'N/A')}"
        
    return response_text


# ✅ 테스트 CLI 실행
if __name__ == "__main__":
    print("\n💬 추천을 시작합니다. (종료하려면 'exit' 입력)")
    while True:
        prompt = input("👤 당신: ")
        if prompt.lower() == "exit":
            print("🤖 추천을 종료합니다.")
            break
        
        response = find_similar_programs(prompt)
        print("🤖 AI 추천:", response)