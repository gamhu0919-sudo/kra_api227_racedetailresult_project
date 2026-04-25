"""
evaluation_view.py
──────────────────────────────
모델 성능 요약, 모델 vs 단순 규칙 비교, 오류(오분류) 분석을 렌더링하는 View 모듈
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from . import utils

def render_rule_comparison(base_test_races, lgbm_test_races, race_id, is_past_mode=True):
    """
    동일 경주(race_id)에 대해 LightGBM 모델과 5개 단순 룰 엔진의 Top3 선택을 비교합니다.
    """
    st.subheader("⚖️ 단순 규칙과 모델(LightGBM) 판단 비교")
    
    b_df = base_test_races[base_test_races['race_id'] == race_id]
    l_df = lgbm_test_races[lgbm_test_races['race_id'] == race_id]
    
    if b_df.empty or l_df.empty:
        st.warning("선택된 경주에 대한 비교 데이터가 없습니다.")
        return

    # 각 방식별 예측 Top3 마번 취합
    picks = {}
    
    # 1. LGBM
    picks['LightGBM 예측 Top3'] = list(l_df[l_df['pred_is_top3'] == 1]['pthrHrno'].astype(str))
    
    # 2. 규칙들 (04_train_baseline_rules 에서 pred_rule1 ~ 5)
    picks['레이팅 상위 3'] = list(b_df[b_df['pred_rule1'] == 1]['pthrHrno'].astype(str))
    picks['말 평균순위 상위 3'] = list(b_df[b_df['pred_rule2'] == 1]['pthrHrno'].astype(str))
    picks['기수 승률 상위 3'] = list(b_df[b_df['pred_rule3'] == 1]['pthrHrno'].astype(str))
    picks['복합 규칙 Top3'] = list(b_df[b_df['pred_rule5'] == 1]['pthrHrno'].astype(str))
    
    actual_top3 = []
    if is_past_mode and 'target_is_top3' in b_df.columns:
        actual_top3 = list(b_df[b_df['target_is_top3'] == 1]['pthrHrno'].astype(str))
        
    comp_list = []
    for method, p in picks.items():
        hit_cnt = len(set(p).intersection(set(actual_top3))) if actual_top3 else "-"
        # 원본에서 실제 우승마 번호를 안다면 hit 여부를 판별하나 target_is_top3 만 있다면 
        # 우승마 특정이 어려우므로 겹친 수로 대체
        comp_list.append({
            "비교 방식": method,
            "예측된 말 번호 리스트": ", ".join(p),
            "실제 Top3와 겹친 수": hit_cnt if is_past_mode else "비공개 (운영모드)"
        })
        
    c_df = pd.DataFrame(comp_list)
    st.dataframe(c_df, use_container_width=True)
    
    if is_past_mode:
        st.info(f"🏁 **실제 해당 경주의 Top3 마번**: {', '.join(actual_top3)}")
        st.write("모델이 단순 사람의 직관(규칙)과 얼마나 다른 유연한 선택을 했는지 확인할 수 있습니다.")

def render_model_performance(eval_df):
    """모델별 전반적 성능 비교표 및 차트"""
    st.subheader("📊 모델 및 전략별 성능 요약 (Test Set 기준)")
    
    if eval_df.empty:
        st.warning("성능 평가 테이블이 존재하지 않습니다.")
        return
        
    # 용어 친화적 변경 적용 대상 컬럼 교체
    eval_df_renamed = eval_df.copy()
    
    rename_cols = {
        'method': '비교 방식',
        'test_race_count': 'Test 경주 수',
        'precision_at_3': '예측 3마리 중 실제 Top3 적중 비율(%)',
        'hit_at_3': '실제 1위마를 예측 Top3 안에 포함한 비율(%)',
        'avg_correct_top3_count': '평균 적중 수 (개)',
        'ndcg_at_3': 'NDCG@3'
    }
    
    eval_df_renamed = eval_df_renamed.rename(columns=rename_cols)
    
    st.dataframe(eval_df_renamed, use_container_width=True)
    
    # Plotly 시각화
    st.markdown("#### 핵심 평가 지표 시각화")
    fig1 = px.bar(eval_df, x='precision_at_3', y='method', orientation='h', 
                  title="Precision@3 비교 (예측 3마리 중 Top3 적중률 %)", color='precision_at_3')
    fig1.update_yaxes(categoryorder='total ascending')
    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = px.bar(eval_df, x='hit_at_3', y='method', orientation='h', 
                  title="Hit@3 비교 (우승마 포함 비율 %)", color='hit_at_3')
    fig2.update_yaxes(categoryorder='total ascending')
    st.plotly_chart(fig2, use_container_width=True)

def render_error_analysis(good_df, bad_df, dist_df, cls_df):
    """Test 데이터 기준 오분류 분석 요약"""
    st.subheader("🎯 예측 오류 및 약점 분석")
    st.markdown("과거 Test 데이터 기준으로 모델이 잘 맞춘 환경과 어려워하는 환경을 분석합니다.")
    
    empty_all = good_df.empty and bad_df.empty and dist_df.empty and cls_df.empty
    if empty_all:
        st.info("오류 분석 파일 없음")
        return
        
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🏆 Good Races (우수 적중 경주 수)", len(good_df))
    with col2:
        st.metric("💀 Bad Races (완전 오답 경주 수)", len(bad_df))
        
    st.markdown("---")
    st.markdown("#### 특정 조건별 성능")
    
    c1, c2 = st.columns(2)
    with c1:
        if not dist_df.empty and 'fe_race_dist' in dist_df.columns:
            fig_d = px.bar(dist_df, x='fe_race_dist', y='hit_ratio', text_auto='.1f',
                           title="거리별 Hit@3 성능 편차")
            st.plotly_chart(fig_d, use_container_width=True)
            
    with c2:
        if not cls_df.empty and 'cndRaceClas' in cls_df.columns:
            fig_c = px.bar(cls_df, x='cndRaceClas', y='hit_ratio', text_auto='.1f',
                           title="경주 등급별 Hit@3 성능 편차")
            st.plotly_chart(fig_c, use_container_width=True)
    
    st.caption("※ 표본 수(경주 수)가 적을수록 통계적 변동성이 클 수 있습니다.")
