import os
import sys
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from datetime import datetime

# 1. 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
STAGE03_ROOT = os.path.join(ROOT_DIR, "kra_race_prediction_stage03_top3_modeling")
STAGE03_SRC = os.path.join(STAGE03_ROOT, "src")

# 학습 시 사용된 config 모듈 임포트 (경로 보정 필요)
sys.path.insert(0, STAGE03_SRC)
import config

DATA_TEMPLATE_PATH = os.path.join(BASE_DIR, "data", "template", "next_race_template.csv")
DATA_REF_DIR = os.path.join(BASE_DIR, "data", "reference")
MODEL_PATH = os.path.join(STAGE03_ROOT, "models", "lightgbm_top3_baseline.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "output", "next_race_predictions.csv")

# 2. 모델 학습 시 사용된 피처 리스트 (중요도 순서 무관하게 전체 리스트 확보 필요)
# lightgbm_feature_importance.csv 기반으로 추출된 최종 피처셋
REQUIRED_FEATURES = [
    'horse_avg_rank_rank_in_race', 'fe_horse_cum_avg_rk', 'jockey_top3_rate_rank_in_race', 
    'weight_zscore_in_race', 'fe_trar_cum_win_rate', 'jockey_winrate_rank_in_race', 
    'rsutTrckStus', 'fe_jcky_cum_top3_rate', 'horse_winrate_rank_in_race', 
    'rating_zscore_in_race', 'fe_weight_diff', 'fe_horse_weight', 'fe_horse_race_count', 
    'trainer_winrate_rank_in_race', 'fe_jcky_cum_win_rate', 'pthrGtno', 'rating_rank_in_race', 
    'rating_pct_rank_in_race', 'fe_ratg_per_weight', 'fe_horse_cum_win_rate', 
    'weight_rank_in_race', 'horse_experience_rank_in_race', 'cndRatg', 'pthrBurdWgt', 
    'pthrRatg', 'cndRaceClas', 'fe_month', 'fe_track_humidity', 'cndGndr', 
    'fe_race_dist', 'fe_season', 'cndAg', 'rsutWetr', 'hrmJckyAlw', 'cndBurdGb'
]

def generate_relative_features(df):
    """Stage 03과 동일한 경주 내 상대 피처 생성 로직"""
    rules = {
        "rating_rank_in_race"         : ("pthrRatg", False),
        "rating_pct_rank_in_race"     : ("pthrRatg", False, "pct"),
        "rating_zscore_in_race"       : ("pthrRatg", None, "zscore"),
        "weight_rank_in_race"         : ("pthrBurdWgt", False),
        "weight_zscore_in_race"       : ("pthrBurdWgt", None, "zscore"),
        "jockey_winrate_rank_in_race" : ("fe_jcky_cum_win_rate", False),
        "jockey_top3_rate_rank_in_race":("fe_jcky_cum_top3_rate", False),
        "trainer_winrate_rank_in_race": ("fe_trar_cum_win_rate", False),
        "horse_avg_rank_rank_in_race" : ("fe_horse_cum_avg_rk", True), 
        "horse_winrate_rank_in_race"  : ("fe_horse_cum_win_rate", False),
        "horse_experience_rank_in_race": ("fe_horse_race_count", False),
    }
    for col, rule_args in rules.items():
        base_col = rule_args[0]
        direction = rule_args[1]
        t = rule_args[2] if len(rule_args) > 2 else "rank"
        if t == "rank":
            df[col] = df.groupby("race_id")[base_col].transform(lambda x: x.rank(method="min", ascending=direction))
        elif t == "pct":
            df[col] = df.groupby("race_id")[base_col].transform(lambda x: x.rank(method="average", ascending=direction, pct=True))
        elif t == "zscore":
            df[col] = df.groupby("race_id")[base_col].transform(lambda x: (x - x.mean())/x.std() if x.std() > 0 else 0)
            df[col] = df[col].fillna(0.0)
    return df

def main():
    if not os.path.exists(DATA_TEMPLATE_PATH):
        print(f"Error: Template not found at {DATA_TEMPLATE_PATH}")
        return

    # 1. 템플릿 로드 (사용자가 기입한 마체중, 날씨, 주로상태 포함)
    print("Loading templates and reference data...")
    df = pd.read_csv(DATA_TEMPLATE_PATH, dtype={'hrmJckyId': str, 'hrmTrarId': str}, encoding="utf-8-sig")
    df['schdRaceDt'] = pd.to_datetime(df['schdRaceDt'])
    
    # 2. 기준 스탯 결합 (말, 기수, 조교사 최신 누적 정보)
    horse_ref = pd.read_csv(os.path.join(DATA_REF_DIR, "horse_latest_stats.csv"), encoding="utf-8-sig")
    jcky_ref = pd.read_csv(os.path.join(DATA_REF_DIR, "jockey_latest_stats.csv"), dtype={'hrmJckyId': str}, encoding="utf-8-sig")
    trar_ref = pd.read_csv(os.path.join(DATA_REF_DIR, "trainer_latest_stats.csv"), dtype={'hrmTrarId': str}, encoding="utf-8-sig")

    df = df.merge(horse_ref, on='pthrHrno', how='left')
    df = df.merge(jcky_ref, on='hrmJckyId', how='left')
    df = df.merge(trar_ref, on='hrmTrarId', how='left')

    # 3. 누락된 피처 가공
    # fe_weight_diff: 사용자 입력 체중 - 과거 마지막 체중
    df['fe_weight_diff'] = df['fe_horse_weight'] - df['last_horse_weight']
    df['fe_weight_diff'] = df['fe_weight_diff'].fillna(0.0)
    
    # fe_ratg_per_weight: 레이팅 / 부담중량
    df['fe_ratg_per_weight'] = df['pthrRatg'] / df['pthrBurdWgt']
    
    # 날짜 관련 피처
    df['fe_month'] = df['schdRaceDt'].dt.month
    df['fe_season'] = (df['fe_month'] % 12 // 3) # 0:겨울, 1:봄, 2:여름, 3:가을
    
    # 누락된 신규 말/기수/조교사 스탯 채우기 (전체 평균 또는 0)
    fill_cols = [
        'fe_horse_cum_win_rate', 'fe_horse_cum_avg_rk', 'fe_horse_race_count',
        'fe_jcky_cum_win_rate', 'fe_jcky_cum_top3_rate', 'fe_trar_cum_win_rate'
    ]
    for c in fill_cols:
        if c == 'fe_horse_cum_avg_rk':
            df[c] = df[c].fillna(10.0) # 기본 10등 수준
        else:
            df[c] = df[c].fillna(0.0)

    # 4. 상대 피처 생성
    print("Generating relative features...")
    df = generate_relative_features(df)

    # 5. 모델 로드 및 추론
    print(f"Loading model from {MODEL_PATH}...")
    lgb_model = joblib.load(MODEL_PATH)
    
    # 데이터 타입 맞추기 (범주형)
    X_inference = df[REQUIRED_FEATURES].copy()
    for col in X_inference.columns:
        if not pd.api.types.is_numeric_dtype(X_inference[col]):
            X_inference[col] = X_inference[col].astype('category')
            
    # 예측 (Top 3 여부 확률)
    probs = lgb_model.predict_proba(X_inference)[:, 1]
    df['top3_prob'] = probs
    
    # 6. 결과 정리: 경주별 확률 순 정렬 및 예상 순위
    df['pred_rank'] = df.groupby('race_id')['top3_prob'].rank(ascending=False, method='min')
    
    # 최종 결과물 컬럼 선별
    final_cols = [
        'race_id', 'schdRaceDt', 'schdRaceNo', 'pthrGtno', 'pthrHrnm', 'hrmJckyNm', 
        'top3_prob', 'pred_rank', 'fe_horse_weight', 'rsutWetr', 'rsutTrckStus'
    ]
    result_df = df[final_cols].sort_values(['race_id', 'pred_rank'])
    
    # 저장
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Inference results saved to {OUTPUT_PATH}")
    print(result_df.head(15))

if __name__ == "__main__":
    main()
