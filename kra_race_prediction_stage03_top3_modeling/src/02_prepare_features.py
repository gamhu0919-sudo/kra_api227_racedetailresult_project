"""
02_prepare_features.py
──────────────────────────────
결측치를 채우고 모델 학습을 위한 최종 피처 조합 마련 
(설계된 피처들의 누락 여부 검증 및 보완)
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

config.log_step(2, "피처 전처리 및 결측치 핸들링")

TARGET_FEATURES = [
    # 5.1 말 관련 피처
    "pthrRatg", "pthrBurdWgt", "fe_ratg_per_weight", "fe_horse_weight", 
    "fe_weight_diff", "fe_horse_cum_win_rate", "fe_horse_cum_avg_rk", "fe_horse_race_count",
    # 5.2 기수 관련 피처
    "fe_jcky_cum_win_rate", "fe_jcky_cum_top3_rate", "hrmJckyAlw",
    # 5.3 조교사 관련 피처
    "fe_trar_cum_win_rate",
    # 5.4 경주 조건 피처
    "cndRaceClas", "fe_race_dist", "cndBurdGb", "cndRatg", "cndGndr", "cndAg",
    # 5.5 환경 피처
    "rsutTrckStus", "rsutWetr", "fe_track_humidity", "fe_month", "fe_season",
    # 5.6 출전 위치
    "pthrGtno",
    # 5.7 경주 내 상대 피처
    "rating_rank_in_race", "rating_pct_rank_in_race", "rating_zscore_in_race", 
    "weight_rank_in_race", "weight_zscore_in_race", "jockey_winrate_rank_in_race", 
    "jockey_top3_rate_rank_in_race", "trainer_winrate_rank_in_race", 
    "horse_avg_rank_rank_in_race", "horse_winrate_rank_in_race", "horse_experience_rank_in_race"
]

def generate_relative_features(df):
    """
    EDA 시 생성되었어야 할 상대 피처가 없을 경우 방어적으로 로컬 생성.
    주의: 원천 피처가 있으면 rank(), zscore 계산
    """
    config.log("누락된 상대 피처에 대해 방어적 생성을 시작합니다.")

    # 베이스 피처 - (새칼럼, 비교방향: False면 높을수록 순위높은것(1등), True면 낮을수록 1등)
    rules = {
        "rating_rank_in_race"         : ("pthrRatg", False),
        "rating_pct_rank_in_race"     : ("pthrRatg", False, "pct"),
        "rating_zscore_in_race"       : ("pthrRatg", None, "zscore"),
        "weight_rank_in_race"         : ("pthrBurdWgt", False),
        "weight_zscore_in_race"       : ("pthrBurdWgt", None, "zscore"),
        "jockey_winrate_rank_in_race" : ("fe_jcky_cum_win_rate", False),
        "jockey_top3_rate_rank_in_race":("fe_jcky_cum_top3_rate", False),
        "trainer_winrate_rank_in_race": ("fe_trar_cum_win_rate", False),
        "horse_avg_rank_rank_in_race" : ("fe_horse_cum_avg_rk", True), # 낮을수록 순위 높음
        "horse_winrate_rank_in_race"  : ("fe_horse_cum_win_rate", False),
        "horse_experience_rank_in_race": ("fe_horse_race_count", False),
    }

    for col, rule_args in rules.items():
        if col in df.columns:
            continue # 이미 있으면 스킵
            
        base_col = rule_args[0]
        if base_col not in df.columns:
            config.log(f"  [경고] {base_col} 원천이 없어서 {col}을 생성할 수 없음.")
            continue
            
        config.log(f"  새 상대피처 생성: {col}")
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
    in_path = os.path.join(config.DATA_PROCESSED, "modeling_data_validated.csv")
    df = pd.read_csv(in_path, encoding="utf-8-sig", low_memory=False)

    # 파생 피처 결측시 방어 생성
    df = generate_relative_features(df)

    # 요구사항 컬럼들의 존재 여부 확인
    available_features = []
    for col in TARGET_FEATURES:
        if col in df.columns:
            available_features.append(col)
        else:
            config.log(f"  [누락] 요구된 피처 '{col}' 가 원본 데이터에 없습니다.")

    # 식별자와 타겟 컬럼 항상 유지
    keep_cols = ["race_id", "schdRaceDt", "pthrHrno", config.TARGET_COL]
    
    # 1. 사용할 컬럼만 필터링 
    df = df[keep_cols + available_features].copy()
    config.log(f"선택된 최종 피처 수: {len(available_features)}개")

    # 2. 결측치 처리 (Imputation)
    missing_info = []

    for col in available_features:
        missing_cnt = df[col].isna().sum()
        dtype = str(df[col].dtype)
        is_num = pd.api.types.is_numeric_dtype(df[col])
        strategy = ""
        impute_val = None

        if missing_cnt > 0:
            if is_num:
                # 숫자형은 결측 지시자 플래그 추가 (01 10 등 정보보존)
                indicator_col = f"{col}_is_na"
                df[indicator_col] = df[col].isna().astype(int)
                
                impute_val = df[col].median()
                df[col] = df[col].fillna(impute_val)
                strategy = f"Median({impute_val:.2f}) + Flag"
            else:
                impute_val = "UNKNOWN"
                df[col] = df[col].fillna(impute_val)
                strategy = "UNKNOWN"
        else:
            strategy = "None (No Missing)"

        missing_info.append({
            "Feature": col,
            "Type": dtype,
            "Missing_Count": missing_cnt,
            "Missing_Ratio(%)": round(missing_cnt / len(df) * 100, 2),
            "Impute_Strategy": strategy
        })

    # 명세서 저장
    missing_df = pd.DataFrame(missing_info)
    summary_path = os.path.join(config.OUT_TABLES, "missing_handling_summary.csv")
    missing_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    config.log(f"결측치 핸들링 리포트 완료: {summary_path}")

    # 최종 Modeling Base 저장
    out_path = os.path.join(config.DATA_PROCESSED, "modeling_data_features.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    config.log(f"최종 피처 준비 데이터 저장 완료: {out_path} (Shape: {df.shape})")

if __name__ == "__main__":
    main()
