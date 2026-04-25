"""
feature_view.py
──────────────────────────────
예측 근거(Feature) 설명 및 변수 중요도를 렌더링하는 View 모듈
"""

import streamlit as st
import pandas as pd
from . import utils

def render_feature_reasoning(selected_horse_row):
    """선택된 1마리 말의 예측 근거(Z-score 및 랭크) 설명 프래그먼트"""
    st.subheader("💡 예측 근거 설명")
    
    if selected_horse_row.empty:
        st.info("말을 선택해주세요.")
        return
        
    s = selected_horse_row.iloc[0]
    
    # 주요 지표 추출
    hr_rank = s.get('horse_avg_rank_rank_in_race', '데이터없음')
    jcky_top3_rank = s.get('jockey_top3_rate_rank_in_race', '데이터없음')
    jcky_win_rank = s.get('jockey_winrate_rank_in_race', '데이터없음')
    trar_win_rank = s.get('trainer_winrate_rank_in_race', '데이터없음')
    
    ratg_z = s.get('rating_zscore_in_race', 0.0)
    wgt_z = s.get('weight_zscore_in_race', 0.0)
    prob = s.get('pred_top3_prob', 0.0) * 100

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**경주 내 상대 순위(Top)**")
        st.write(f"- 🐎 말 과거 평균성적 순위: **{hr_rank}위**")
        st.write(f"- 🏇 기수 Top3율 순위: **{jcky_top3_rank}위**")
        st.write(f"- 🏇 기수 승률 순위: **{jcky_win_rank}위**")
        st.write(f"- 👨‍🏫 조교사 승률 순위: **{trar_win_rank}위**")

    with col2:
        st.markdown("**Z-Score (평균 대비 편차)**")
        st.write(f"- 📈 레이팅 가중치: **{ratg_z:.2f}** (높을수록 체급 우위)")
        st.write(f"- ⚖️ 부담중량 페널티: **{wgt_z:.2f}** (높을수록 무거움)")
        st.markdown(f"#### 🎯 최종 예측 확률: {prob:.1f}%")

    st.markdown("---")
    # 초보자 친화적 설명 생성
    reasoning_text = []
    
    if isinstance(hr_rank, (int, float)) and hr_rank <= 3:
        reasoning_text.append("이 말은 같은 경주 출전마 중 **과거 성적이 최상위권**에 속합니다.")
    else:
        reasoning_text.append("이 말은 과거 성적 면에서는 평범하거나 다소 부족한 편입니다.")

    if isinstance(jcky_top3_rank, (int, float)) and jcky_top3_rank <= 3:
        reasoning_text.append(f"또한, 기승하는 기수의 입상률이 전체 **{int(jcky_top3_rank)}위**로 매우 우수하여 긍정적인 평가를 받았습니다.")
        
    if isinstance(ratg_z, (int, float)) and ratg_z > 0.5:
        reasoning_text.append("레이팅(능력치)도 해당 경주의 평균 출전마들보다 두각을 나타내고 있습니다.")
        
    if prob >= 40.0:
        reasoning_text.append("👉 **종합적으로 모델은 이 말의 Top3 입상 가능성을 상당히 높게 평가했습니다.**")
    elif prob >= 20.0:
        reasoning_text.append("👉 **평이한 조건 속에서 다크호스로 분류될 기회가 있습니다.**")
    else:
        reasoning_text.append("👉 **현재 조건상 객관적인 지표에서는 입상 확률이 낮게 평가되었습니다.**")

    st.success(" ".join(reasoning_text))

def render_feature_importance(fi_df):
    """변수 중요도 Top20 표시"""
    st.subheader("🔍 변수 중요도 (Feature Importance)")
    st.markdown("모델이 승부를 가르는 데 가장 중요하게 바라본 데이터 Top 20입니다.")
    
    if fi_df.empty:
        st.warning("변수 중요도 데이터를 불러올 수 없습니다.")
        return
        
    top20 = fi_df.head(20).copy()
    
    # 이해하기 쉬운 설명 매핑 (요구사항)
    desc_dict = {
        "horse_avg_rank_rank_in_race": "같은 경주 안에서 말의 과거 평균순위가 몇 번째로 좋은지",
        "fe_horse_cum_avg_rk": "말의 과거 평균 순위",
        "jockey_top3_rate_rank_in_race": "같은 경주 안에서 기수의 Top3율 순위",
        "weight_zscore_in_race": "같은 경주 안에서 부담중량이 평균보다 얼마나 높은지",
        "rating_zscore_in_race": "같은 경주 안에서 레이팅이 평균보다 얼마나 높은지",
        "fe_horse_race_count": "말의 총 출전 횟수 (경험치)",
        "pthrRatg": "말의 절대 레이팅 점수",
        "pthrBurdWgt": "말이 짊어질 절대 부담 중량",
        "fe_jcky_cum_win_rate": "기수의 절대적 누적 승률",
        "fe_trar_cum_win_rate": "조교사의 절대적 누적 승률",
    }
    
    top20['쉬운 설명'] = top20['feature'].map(lambda x: desc_dict.get(x, "모델의 세부 분석 변수"))
    top20 = top20.rename(columns={'feature':'변수명', 'importance':'중요도 점수'})
    
    st.dataframe(top20[['변수명', '쉬운 설명', '중요도 점수']], use_container_width=True)
