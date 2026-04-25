"""
utils.py
──────────────────────────────
Streamlit 대시보드 공통 유틸리티 및 경로 설정
"""

import os
import streamlit as st

# ──────────────────────────────────────────────
# 1. 파일 경로 참조 헬퍼
# ──────────────────────────────────────────────
# 현재 스크립트 위치: src/
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# 스테이지04 루트
STAGE04_DIR = os.path.dirname(SRC_DIR)
# 프로젝트 루트
PROJECT_ROOT = os.path.dirname(STAGE04_DIR)
# 이전 스테이지 연동 위치
STAGE03_DIR = os.path.join(PROJECT_ROOT, "kra_race_prediction_stage03_top3_modeling")
STAGE05_DIR = os.path.join(PROJECT_ROOT, "kra_race_prediction_stage05_inference_pipeline")

PATH_MODEL = os.path.join(STAGE03_DIR, "models", "lightgbm_top3_baseline.pkl")
PATH_LGBM_PRED = os.path.join(STAGE03_DIR, "data", "predictions", "lightgbm_top3_predictions.csv")
PATH_NEXT_PRED = os.path.join(STAGE05_DIR, "data", "output", "next_race_predictions.csv")
PATH_BASE_PRED = os.path.join(STAGE03_DIR, "data", "predictions", "baseline_rule_predictions.csv")
PATH_MOD_READY = os.path.join(STAGE03_DIR, "data", "modeling", "modeling_data_ready.csv")
PATH_CMP_TBL = os.path.join(STAGE03_DIR, "outputs", "tables", "model_comparison_table.csv")
PATH_FI = os.path.join(STAGE03_DIR, "outputs", "tables", "lightgbm_feature_importance.csv")

# 에러 분석 파일 경로들
PATH_ERR_GOOD = os.path.join(STAGE03_DIR, "outputs", "tables", "good_prediction_races.csv")
PATH_ERR_BAD = os.path.join(STAGE03_DIR, "outputs", "tables", "bad_prediction_races.csv")
PATH_ERR_DIST = os.path.join(STAGE03_DIR, "outputs", "tables", "error_analysis_by_distance.csv")
PATH_ERR_CLS = os.path.join(STAGE03_DIR, "outputs", "tables", "error_analysis_by_class.csv")


# ──────────────────────────────────────────────
# 2. 용어 매핑 및 포매터
# ──────────────────────────────────────────────
TERMINOLOGY_MAP = {
    # 내부 피처 / 컬럼명 -> 쉬운 표시 용어
    "pred_top3_prob": "Top3 예상 확률",
    "pred_rank_in_race": "모델 예상 순위",
    "pred_is_top3": "모델 Top3 선택",
    "Precision@3": "예측 3마리 중 실제 Top3 적중 비율",
    "Hit@3": "실제 1위마를 예측 Top3 안에 포함한 비율",
    
    # 주요 Feature
    "horse_avg_rank_rank_in_race": "경주 내 과거 성적 순위",
    "rating_zscore_in_race": "경주 내 레이팅 우위",
    "weight_zscore_in_race": "경주 내 부담중량 차이",
    "fe_horse_cum_avg_rk": "말의 과거 평균 순위",
    "jockey_top3_rate_rank_in_race": "같은 경주 안에서 기수의 Top3율 순위",
    
    # 원본 메타 데이터들
    "pthrHrno": "출전번호",
    "prdctnRank": "예측순위",
    "pthrRatg": "레이팅",
    "pthrBurdWgt": "부담중량",
    "fe_jcky_cum_top3_rate": "기수 누적 Top3율",
    "fe_trar_cum_win_rate": "조교사 누적 승률"
}

def translate_term(internal_term: str) -> str:
    """내부 용어가 매핑 테이블에 존재하면 친화적 용어로 반환"""
    return TERMINOLOGY_MAP.get(internal_term, internal_term)

def apply_friendly_columns(df):
    """데이터프레임의 컬럼명을 사용자 친화적으로 변경"""
    return df.rename(columns=TERMINOLOGY_MAP)

def render_warning_disclaimer():
    """앱 전반에 노출해야 할 보증 제한 및 경고 문구"""
    st.warning("⚠️ **주의**: 본 대시보드는 경주 전 데이터 기반 Top3 진입 가능성을 분석하기 위한 프로토타입입니다. "
               "실제 경주 결과를 보장하지 않으며, 도박 또는 베팅 권유 목적이 아닙니다. 참고용 예측 결과로만 활용해 주십시오.")
