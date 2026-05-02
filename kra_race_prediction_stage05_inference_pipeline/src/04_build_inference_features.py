import os
import pandas as pd
import numpy as np
import inference_config as cfg

def main():
    cfg.log("기초 피처 생성 및 기준 데이터 조인을 시작합니다.")
    
    # 1. 입력 데이터 로드
    df = pd.read_csv(cfg.PATH_INPUT_ENTRIES, dtype={'hrmJckyId': str, 'hrmTrarId': str, 'pthrHrno': str}, encoding="utf-8-sig")
    df['schdRaceDt'] = pd.to_datetime(df['schdRaceDt'])

    # 2. 기준 테이블 로드
    horse_ref = pd.read_csv(cfg.PATH_REF_HORSE, dtype={'pthrHrno': str}, encoding="utf-8-sig")
    jcky_ref = pd.read_csv(cfg.PATH_REF_JOCKEY, dtype={'hrmJckyId': str}, encoding="utf-8-sig")
    trar_ref = pd.read_csv(cfg.PATH_REF_TRAINER, dtype={'hrmTrarId': str}, encoding="utf-8-sig")

    # 3. Join
    df = df.merge(horse_ref, on='pthrHrno', how='left')
    df = df.merge(jcky_ref, on='hrmJckyId', how='left')
    df = df.merge(trar_ref, on='hrmTrarId', how='left')

    # 4. Cold-start 플래그 및 기본값 처리
    df['is_new_horse'] = df['fe_horse_race_count'].isna().astype(int)
    df['is_new_jockey'] = df['fe_jcky_cum_win_rate'].isna().astype(int)
    df['is_new_trainer'] = df['fe_trar_cum_win_rate'].isna().astype(int)

    # 마체중 증감 계산
    df['fe_weight_diff'] = df['fe_horse_weight'] - df['last_horse_weight']
    df.loc[df['last_horse_weight'].isna() | (df['fe_horse_weight'] == 0), 'fe_weight_diff'] = 0.0
    
    # 4-1. 휴식일수 및 휴식 유형 계산 (v2 신규)
    if 'last_race_date' in df.columns:
        df['last_race_date'] = pd.to_datetime(df['last_race_date'])
        df['fe_horse_days_since_last_race'] = (df['schdRaceDt'] - df['last_race_date']).dt.days
        # 신마 등 데이터가 없는 경우 처리 (999일 또는 평균값)
        df['fe_horse_days_since_last_race'] = df['fe_horse_days_since_last_race'].fillna(100) 
        
        # 휴식 유형 정의
        def get_rest_type(days):
            if days <= 30: return 'short_rest'
            if days <= 60: return 'normal_rest'
            if days > 60: return 'long_rest'
            return 'unknown_rest'
        df['fe_horse_rest_type'] = df['fe_horse_days_since_last_race'].apply(get_rest_type)
    else:
        df['fe_horse_days_since_last_race'] = 100
        df['fe_horse_rest_type'] = 'unknown_rest'

    # 4-2. 기초 가공 피처
    df['fe_ratg_per_weight'] = df['pthrRatg'] / df['pthrBurdWgt']
    df['fe_month'] = df['schdRaceDt'].dt.month
    df['fe_season'] = (df['fe_month'] % 12 // 3)

    # 4-3. 통계 결측 보완 (v2 피처들 포함)
    # 수치형 피처들 자동 채우기 (fillna(0))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    
    # 특정 피처 예외 처리 (평균 순위 등)
    if 'fe_horse_cum_avg_rk' in df.columns:
        df.loc[df['is_new_horse'] == 1, 'fe_horse_cum_avg_rk'] = 10.0
    if 'fe_horse_recent3_avg_rank' in df.columns:
        df.loc[df['is_new_horse'] == 1, 'fe_horse_recent3_avg_rank'] = 10.0
    if 'fe_horse_recent5_avg_rank' in df.columns:
        df.loc[df['is_new_horse'] == 1, 'fe_horse_recent5_avg_rank'] = 10.0

    # 5. 저장
    df.to_csv(cfg.PATH_FE_BASE, index=False, encoding="utf-8-sig")
    cfg.log(f"기초 피처 데이터 저장 완료: {cfg.PATH_FE_BASE}")

if __name__ == "__main__":
    main()
