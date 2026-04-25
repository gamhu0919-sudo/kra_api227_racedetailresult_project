"""
prediction_service.py
──────────────────────────────
LightGBM 예측 모델 싱글턴 로더 및 추론 파이프라인
"""

import os
import joblib
import pandas as pd
import streamlit as st
from . import utils

@st.cache_resource
def load_model():
    """LightGBM 모델 로드"""
    if os.path.exists(utils.PATH_MODEL):
        model = joblib.load(utils.PATH_MODEL)
        return model
    else:
        st.error(f"모델 파일이 존재하지 않습니다: {utils.PATH_MODEL}")
        return None

def validate_features(df, model_features):
    """모델 입력 피처 누락 여부 지원 검증"""
    missing_cols = [col for col in model_features if col not in df.columns]
    if missing_cols:
        st.error(f"입력 데이터에 다음 필수 피처가 누락되었습니다: {missing_cols}")
        return False

    # 누수 컬럼 확인 (대상 목록은 utils 등에 있으나 여기서 문자열로 필터 검증)
    leakage_cols = [
        "rsutRk", "target_rank", "rsutRaceRcd", "rsutMargin",
        "rsutRkAdmny", "rsutRkPurse", "rsutQnlaPrice", "rsutWinPrice"
    ]
    leaks = [c for c in leakage_cols if c in df.columns]
    if leaks:
        st.warning(f"학습 시 제외되어야 할 데이터(누수 의심 컬럼)가 포함되어 있습니다: {leaks}")
        # 모델 구동엔 영향을 안주지만 경고
        
    return True

def predict_top3(df_race, model):
    """
    특정 race_id 의 출전마 데이터(df_race)를 받아
    pred_top3_prob, pred_rank_in_race, pred_is_top3 열을 산출하여 반환.
    """
    if len(df_race) == 0:
        return df_race

    # 모델이 학습한 피처 구성 로드 (model_features) - LGBM 속성 사용
    model_features = model.feature_name_
    
    if not validate_features(df_race, model_features):
        return None

    # 카테고리 형변환 처리 필요 (만약 문자열 객체로 남아있다면)
    X = df_race[model_features].copy()
    for f in model_features:
        if X[f].dtype == object or X[f].dtype.name == 'string':
            X[f] = X[f].astype('category')
            
    # 확률 산출
    probs = model.predict_proba(X)[:, 1]
    
    result_df = df_race.copy()
    result_df['pred_top3_prob'] = probs
    
    # 확률 기준 내림차순 랭킹
    result_df['pred_rank_in_race'] = result_df['pred_top3_prob'].rank(ascending=False, method='min')
    
    # 0.5가 아니라 랭크 기준 3위 이내면 판별
    result_df['pred_is_top3'] = (result_df['pred_rank_in_race'] <= 3).astype(int)
    
    return result_df
