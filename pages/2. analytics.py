import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud 
import matplotlib.pyplot as plt 
from PIL import Image


st.set_page_config(page_title="지원사업 분석", page_icon="📊", layout="wide")

# --- 데이터 로딩 (home.py와 동일한 캐싱 함수 사용) ---
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['신청시작일자'] = pd.to_datetime(df['신청시작일자'], errors='coerce')
    df['신청종료일자'] = pd.to_datetime(df['신청종료일자'], errors='coerce')
    return df

try:
    df = load_data('./data/programs.csv')
except FileNotFoundError:
    st.error("🚨 'data/programs.csv' 파일을 찾을 수 없습니다.")
    st.stop()

# --- 페이지 제목 ---
st.title("📊 지원사업 트렌드 분석")
st.markdown("정부 지원사업의 분포와 주요 동향을 살펴보세요.")
st.markdown("---")

# --- 1. 통계 요약 ---
st.header("📈 요약 정보")

# 현재 신청 가능한 사업 수 계산
available_now = df[df['신청종료일자'] >= pd.to_datetime('today')].shape[0]
top_department = df['소관부처'].mode()[0] if not df['소관부처'].empty else "N/A"

col1, col2, col3 = st.columns(3)
col1.metric("총 사업 공고 수", f"{df.shape[0]} 건")
col2.metric("현재 신청 가능 사업", f"{available_now} 건")
col3.metric("최다 공고 부처", top_department)

st.markdown("---")

# --- 2. 카테고리별 분포 ---
st.header("📊 카테고리별 분포")

col1, col2 = st.columns(2)

with col1:
    st.subheader("소관부처별 사업 공고 수")
    # '소관부처'별로 카운트하고 상위 10개만 선택
    dept_counts = df['소관부처'].value_counts().nlargest(10)
    fig1 = px.bar(dept_counts, x=dept_counts.index, y=dept_counts.values, 
                  labels={'x':'소관부처', 'y':'공고 수'},
                  color=dept_counts.index)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("지원분야별 사업 공고 수")
    field_counts = df['지원분야'].value_counts().nlargest(10)
    fig2 = px.pie(field_counts, names=field_counts.index, values=field_counts.values,
                  hole=.3) # 도넛 차트 효과
    st.plotly_chart(fig2, use_container_width=True)


# --- 3. 월별 신규 공고 추이 (새로 추가된 부분) ---
st.header("📅 월별 신규 사업 공고 추이")
st.write("'신청시작일자'를 기준으로 월별 신규 공고 등록 건수를 보여줍니다.")

# '신청시작일자'에 유효한 날짜가 있는 데이터만 필터링
df_with_dates = df.dropna(subset=['신청시작일자'])

# 'YYYY-MM' 형식의 '월' 컬럼 생성
df_with_dates['연월'] = df_with_dates['신청시작일자'].dt.to_period('M').astype(str)

# 월별 공고 수 계산 후, 월 순서대로 정렬
monthly_counts = df_with_dates['연월'].value_counts().sort_index()

# Plotly를 사용하여 막대그래프 생성
if not monthly_counts.empty:
    fig_monthly = px.bar(
        monthly_counts,
        x=monthly_counts.index,
        y=monthly_counts.values,
        labels={'x': '연월', 'y': '신규 공고 수'},
        title='월별 신규 사업 공고 등록 건수'
    )
    # x축 레이블이 겹치지 않도록 각도 조절
    fig_monthly.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_monthly, use_container_width=True)
else:
    st.warning("날짜 데이터가 충분하지 않아 월별 추이 그래프를 표시할 수 없습니다.")



# --- 4. 워드클라우드 추가 ---
st.header("🔑 지원내용 핵심 키워드")

# '지원내용' 컬럼의 모든 텍스트를 하나의 문자열로 합치기
# .dropna()로 비어있는 값은 제외
text = ' '.join(df['지원내용'].dropna())
korean_stopwords = set([
    '및', '위한', '으로', '이', '그', '때', '수', '것', '있는', '대한', '할', '있습니다', '있으며',
    '등', '또는', '통해', '하는', '따라', '경우', '내', '기반', '지원', '사업', '위해', '과', '제공',
    '관련', '위치', '대한', '전', '후', '고', '하는', '될', '저', '개', '이번', '안', '데', '요','통한','2025년',
    '최대','기업','등을','지원하는','필요한','지원합니다','지원을','지원으로','지원사업','지원하여','기업의','제품','사업을',
    '중소기업의','활용한','있도록','구축을','소상공인의','사업화를','중소기업이'
])

# 폰트 경로 설정 (2단계에서 준비한 폰트 파일 경로)
font_path = './fonts/GmarketSansTTFMedium.ttf' 
try:
    # 예시: '손' 아이콘 사용
    mask_image_path = './fonts/upgrade2.png' 
    mask_array = np.array(Image.open(mask_image_path))
except FileNotFoundError:
    st.warning(f"마스크 이미지를 찾을 수 없습니다: {mask_image_path}. 기본 사각형으로 표시됩니다.")
    mask_array = None

try:
    # WordCloud 객체 생성 시 width, height, stopwords 파라미터 추가/수정
    wordcloud = WordCloud(
        font_path=font_path,
        width=400,          # 너비 조절 (예: 1000, 원본 800)
        height=400,          # 높이 조절 (예: 500, 원본 400)
        background_color='black',
        stopwords=korean_stopwords, 
        mask=mask_array,
        max_words=80,    
        colormap='autumn'  
    ).generate(text)

    # --- 레이아웃 분할 ---
    # 화면을 2개의 컬럼으로 나누되, 왼쪽이 더 넓게 (2:1 비율)
    col1, col2 = st.columns([2, 1])

    with col1:
        # 워드클라우드 시각화
        fig, ax = plt.subplots(figsize=(3, 3)) # figsize를 WordCloud의 width, height에 맞춰 조절
        fig.patch.set_alpha(0.0) # Matplotlib Figure 배경 투명하게
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

except FileNotFoundError:
    st.error(f"폰트 파일을 찾을 수 없습니다: {font_path}")
    st.warning("프로젝트 내 'fonts' 폴더에 'NanumGothic.ttf' 파일이 있는지 확인하거나, 경로를 수정해주세요.")
except Exception as e:
    st.error(f"워드클라우드를 생성하는 중 오류가 발생했습니다: {e}")
