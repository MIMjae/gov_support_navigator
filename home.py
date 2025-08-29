# home.py

import streamlit as st
import pandas as pd
import numpy as np

# --- 페이지 설정 ---
st.set_page_config(
    page_title="정부 지원사업 정보",
    page_icon="📊",
    layout="wide"
)

# --- 데이터 로딩 (캐싱 적용) ---
@st.cache_data
def load_data(path):
    """CSV 데이터를 로드하고 날짜 컬럼을 datetime으로 변환합니다."""
    df = pd.read_csv(path)
    # '신청종료일자'가 날짜 형식이 아니면 오류가 날 수 있으므로, errors='coerce'로 처리
    df['신청종료일자'] = pd.to_datetime(df['신청종료일자'], errors='coerce')
    return df

try:
    programs_df = load_data('./data/programs.csv')
except FileNotFoundError:
    st.error("🚨 'data/programs.csv' 파일을 찾을 수 없습니다.")
    st.stop()

# --- 사이드바 필터 (기존과 동일) ---
st.sidebar.header("🔍 필터")
# (소관부처, 지원분야, 키워드 필터 로직은 여기에 그대로 둡니다)
departments = programs_df['소관부처'].dropna().unique()
selected_departments = st.sidebar.multiselect('소관부처', options=departments)
fields = programs_df['지원분야'].dropna().unique()
selected_fields = st.sidebar.multiselect('지원분야', options=fields)
keyword = st.sidebar.text_input('공고명 키워드 검색')

# --- 필터링 로직 (기존과 동일) ---
filtered_df = programs_df.copy()
if selected_departments:
    filtered_df = filtered_df[filtered_df['소관부처'].isin(selected_departments)]
if selected_fields:
    filtered_df = filtered_df[filtered_df['지원분야'].isin(selected_fields)]
if keyword:
    filtered_df = filtered_df[filtered_df['공고명'].str.contains(keyword, case=False, na=False)]

# --- 메인 화면 ---
st.title("📊 정부 지원사업 대시보드")
st.markdown("사이드바 필터와 아래 탭을 이용해 원하는 사업 공고를 쉽게 찾아보세요.")
st.markdown("---")

# --- 탭(Tab) 생성 ---
tab1, tab2 = st.tabs(["✅ **신청 가능한 사업**", "종료된 사업"])

# 오늘 날짜를 기준으로 데이터 분리
today = pd.to_datetime('today').normalize() # 시간은 제외하고 날짜만 비교

# '신청종료일자'에 유효한 날짜가 있는 데이터만 필터링
valid_dates_df = filtered_df.dropna(subset=['신청종료일자'])

ongoing_df = valid_dates_df[valid_dates_df['신청종료일자'] >= today]
ended_df = valid_dates_df[valid_dates_df['신청종료일자'] < today]


# --- "신청 가능한 사업" 탭 내용 ---
with tab1:
    st.header(f"현재 신청 가능한 사업: {len(ongoing_df)} 건")
    
    # 신청 마감일이 임박한 순으로 정렬
    ongoing_df = ongoing_df.sort_values(by='신청종료일자', ascending=True)
    
    st.dataframe(
        ongoing_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "공고상세URL": st.column_config.LinkColumn("바로가기", display_text="🔗 Link"),
            "신청종료일자": st.column_config.DateColumn("신청 종료일", format="YYYY-MM-DD"),
        }
    )

# --- "종료된 사업" 탭 내용 ---
with tab2:
    st.header(f"최근에 종료된 사업: {len(ended_df)} 건")
    
    # 최근에 마감된 순으로 정렬
    ended_df = ended_df.sort_values(by='신청종료일자', ascending=False)
    
    st.dataframe(
        ended_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "공고상세URL": st.column_config.LinkColumn("바로가기", display_text="🔗 Link"),
            "신청종료일자": st.column_config.DateColumn("신청 종료일", format="YYYY-MM-DD"),
        }
    )