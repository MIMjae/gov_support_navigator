# create_embeddings.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

MODEL_NAME = 'jhgan/ko-sbert-nli'
DATA_PATH = './data/programs.csv'
EMBEDDING_SAVE_PATH = './models/program_embeddings.npy'
DATA_SAVE_PATH = './models/programs_data.pkl'

# 1. 모델 및 데이터 로드
print("🔍 모델과 데이터를 로드합니다...")
model = SentenceTransformer(MODEL_NAME)
df = pd.read_csv(DATA_PATH)

# 2. 데이터 전처리
print("📊 데이터를 전처리합니다...")
search_columns = ['공고명', '지원분야', '모집 대상', '모집 자격', '지원내용']
df[search_columns] = df[search_columns].fillna('정보 없음')
df['search_text'] = df[search_columns].apply(' '.join, axis=1)
programs_data = df.to_dict('records')

# 3. 임베딩 생성
print("⏳ 텍스트 데이터를 벡터로 변환합니다. (시간이 걸릴 수 있습니다)")
program_texts = [p['search_text'] for p in programs_data]
program_embeddings = model.encode(program_texts, show_progress_bar=True)

# 4. 생성된 임베딩과 데이터 저장
print(f"💾 생성된 임베딩을 '{EMBEDDING_SAVE_PATH}' 파일로 저장합니다.")
np.save(EMBEDDING_SAVE_PATH, program_embeddings)

print(f"💾 전처리된 데이터를 '{DATA_SAVE_PATH}' 파일로 저장합니다.")
with open(DATA_SAVE_PATH, 'wb') as f:
    pickle.dump(programs_data, f)

print("✅ 모든 작업이 완료되었습니다!")