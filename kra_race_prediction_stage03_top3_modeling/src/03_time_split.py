"""
03_time_split.py
──────────────────────────────
경주 일자(schdRaceDt)를 기준으로 Train/Valid/Test 세트를 시간순으로 분할합니다.
동일 race_id가 여러 세트에 나뉘어 들어가지 않도록 보호합니다.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

config.log_step(3, "시간 기반(Time-based) 데이터 분할")

def main():
    in_path = os.path.join(config.DATA_PROCESSED, "modeling_data_features.csv")
    df = pd.read_csv(in_path, encoding="utf-8-sig", low_memory=False)

    # schdRaceDt datetime 변환 및 정렬
    df['schdRaceDt'] = pd.to_datetime(df['schdRaceDt'])
    
    # race_id 별 가장 이른 일자(동일 경주 내에선 일자가 모두 동일하지만 만약을 대비해)를 구해 정렬
    race_dates = df.groupby('race_id')['schdRaceDt'].min().sort_values().reset_index()
    total_races = len(race_dates)
    
    # 70% / 15% / 15% 분할 지점 계산
    train_cut = int(total_races * 0.70)
    valid_cut = int(total_races * 0.85)

    train_races = set(race_dates['race_id'].iloc[:train_cut])
    valid_races = set(race_dates['race_id'].iloc[train_cut:valid_cut])
    test_races  = set(race_dates['race_id'].iloc[valid_cut:])

    def assign_split(rid):
        if rid in train_races: return 'train'
        elif rid in valid_races: return 'valid'
        elif rid in test_races: return 'test'
        return 'unknown'

    df['split_group'] = df['race_id'].apply(assign_split)

    # 결과 검증 및 요약
    split_info = []
    for g in ['train', 'valid', 'test']:
        sub = df[df['split_group'] == g]
        if not sub.empty:
            races = sub['race_id'].nunique()
            dates_min = sub['schdRaceDt'].min().strftime('%Y-%m-%d')
            dates_max = sub['schdRaceDt'].max().strftime('%Y-%m-%d')
            entries = len(sub)
            top3_rate = round(sub[config.TARGET_COL].mean() * 100, 2)
        else:
            races, dates_min, dates_max, entries, top3_rate = 0, "-", "-", 0, 0.0

        split_info.append({
            "Split": g.upper(),
            "Races Count": races,
            "Entries Count": entries,
            "Date Start": dates_min,
            "Date End": dates_max,
            "Top3 Ratio(%)": top3_rate
        })
        config.log(f"[{g.upper()}] Races:{races:,} | {dates_min} ~ {dates_max} | Top3:{top3_rate}%")

    # 교집합 검사 (동일 race_id가 나뉘었는지)
    overlap_tv = len(train_races.intersection(valid_races))
    overlap_vt = len(valid_races.intersection(test_races))
    overlap_tt = len(train_races.intersection(test_races))
    
    if overlap_tv > 0 or overlap_vt > 0 or overlap_tt > 0:
        config.log("[FATAL ERROR] Split group 교집합 존재! (Leakage)")
        sys.exit(1)
    else:
        config.log("Train/Valid/Test 간 race_id 중첩 없음 확인 완료.")

    # 저장
    summary_df = pd.DataFrame(split_info)
    summary_path = os.path.join(config.OUT_TABLES, "train_valid_test_split_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    
    out_path = os.path.join(config.DATA_MODELING, "modeling_data_ready.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    config.log(f"분할 완료! 모델링 데이터셋 최종 저장: {out_path}")

if __name__ == "__main__":
    main()
