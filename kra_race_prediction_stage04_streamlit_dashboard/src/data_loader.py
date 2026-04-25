"""
data_loader.py
──────────────────────────────
Streamlit 대시보드 컴포넌트 렌더링에 필요한 각종 데이터 세트 및 CSV 캐싱 로드
"""

import os
import pandas as pd
import streamlit as st
from . import utils

@st.cache_data
def load_all_datasets():
    """앱 렌더링에 필요한 모든 핵심 데이터를 로드 (캐싱 보장)"""
    data = {}
    
    # 1. 모델링 기준 전체 데이터 (원천 및 피처/메타 병합용)
    if os.path.exists(utils.PATH_MOD_READY):
        df_mod = pd.read_csv(utils.PATH_MOD_READY, encoding="utf-8-sig", low_memory=False)
        
        # 모델의 LightGBM categorical_feature 매칭을 맞추기 위해 
        # 전체 데이터 기준(Categories index 보존)으로 범주형 미리 캐스팅
        for col in df_mod.columns:
            if not pd.api.types.is_numeric_dtype(df_mod[col]) and col not in ['race_id', 'schdRaceDt', 'split_group']:
                df_mod[col] = df_mod[col].astype('category')
                
        data["modeling_ready"] = df_mod
    else:
        st.error(f"모델링 인풋 데이터 누락: {utils.PATH_MOD_READY}")
        
    # 2. LightGBM 예측 결과
    if os.path.exists(utils.PATH_LGBM_PRED):
        data["lgbm_pred"] = pd.read_csv(utils.PATH_LGBM_PRED, encoding="utf-8-sig")
    else:
        st.error("LightGBM 예측 데이터 누락")
        
    # 3. 룰 엔진 예측 결과
    if os.path.exists(utils.PATH_BASE_PRED):
        data["rule_pred"] = pd.read_csv(utils.PATH_BASE_PRED, encoding="utf-8-sig")
    else:
        st.error("규칙 베이스라인 예측 데이터 누락")
        
    # 4. 성능 지표표
    if os.path.exists(utils.PATH_CMP_TBL):
        data["eval_table"] = pd.read_csv(utils.PATH_CMP_TBL, encoding="utf-8-sig")
    else:
        st.warning("모델 평가 테이블 누락")
        data["eval_table"] = pd.DataFrame()

    # 5. 피처 중요도
    if os.path.exists(utils.PATH_FI):
        data["feature_importances"] = pd.read_csv(utils.PATH_FI, encoding="utf-8-sig")
    else:
        st.warning("LightGBM 피처 중요도 누락")
        data["feature_importances"] = pd.DataFrame()

    # 6. Error Analysis 데이터 4종
    err_paths = {
        "err_good": utils.PATH_ERR_GOOD,
        "err_bad": utils.PATH_ERR_BAD,
        "err_dist": utils.PATH_ERR_DIST,
        "err_cls": utils.PATH_ERR_CLS
    }
    
    for key, path in err_paths.items():
        if os.path.exists(path):
            data[key] = pd.read_csv(path, encoding="utf-8-sig")
        else:
            data[key] = pd.DataFrame() # 빈 DF로 처리하여 UI에서 분기

    # 부가 작업: lgbm_pred에 target_rank 등 답안지(순위)가 있다면 추후 join할 수 있게 하지만
    # 이전 Stage에서 modeling_ready의 기반 원본에서 target_rank를 찾아오면 좋습니다.
    # 단, 이번 과업에서는 모델 데이터에 답(is_top3)이 있으니 실제 순위 열은 없더라도
    # "is_top3" 를 정답지로 표시합니다.
    
    return data
