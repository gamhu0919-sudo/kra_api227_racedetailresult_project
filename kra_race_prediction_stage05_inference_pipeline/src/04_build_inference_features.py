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
    # last_horse_weight가 없거나 현재 fe_horse_weight가 0이면 0으로 처리
    df['fe_weight_diff'] = df['fe_horse_weight'] - df['last_horse_weight']
    df.loc[df['last_horse_weight'].isna() | (df['fe_horse_weight'] == 0), 'fe_weight_diff'] = 0.0
    
    # 기초 가공 피처
    df['fe_ratg_per_weight'] = df['pthrRatg'] / df['pthrBurdWgt']
    df['fe_month'] = df['schdRaceDt'].dt.month
    df['fe_season'] = (df['fe_month'] % 12 // 3)

    # 통계 결측 보완 (Stage03 기준 적용)
    df['fe_horse_cum_win_rate'] = df['fe_horse_cum_win_rate'].fillna(0.0)
    df['fe_horse_cum_avg_rk'] = df['fe_horse_cum_avg_rk'].fillna(10.0)
    df['fe_horse_race_count'] = df['fe_horse_race_count'].fillna(0)
    df['fe_jcky_cum_win_rate'] = df['fe_jcky_cum_win_rate'].fillna(0.0)
    df['fe_jcky_cum_top3_rate'] = df['fe_jcky_cum_top3_rate'].fillna(0.0)
    df['fe_trar_cum_win_rate'] = df['fe_trar_cum_win_rate'].fillna(0.0)

    # 5. 저장
    df.to_csv(cfg.PATH_FE_BASE, index=False, encoding="utf-8-sig")
    cfg.log(f"기초 피처 데이터 저장 완료: {cfg.PATH_FE_BASE}")

if __name__ == "__main__":
    main()
