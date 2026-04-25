"""
04_train_baseline_rules.py
──────────────────────────────
단순 규칙(휴리스틱) 베이스라인 예측을 생성합니다. 
생성된 예측은 개별 경주(race_id)별 상위 3위를 지정하는 것으로 처리합니다.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

config.log_step(4, "단순 규칙 기반 베이스라인(Rule Baseline) 예측 생성")

def select_top3_by_score(df, score_col, ascending=False):
    """
    각 race_id 내에서 score_col 기준으로 상위 3위를 선별.
    ascending=True일 경우 값이 낮은 것이 좋은 것 (예: 평균순위).
    """
    if ascending:
        df_sorted = df.sort_values(['race_id', score_col], ascending=[True, True])
    else:
        df_sorted = df.sort_values(['race_id', score_col], ascending=[True, False])
        
    df_sorted['rank_tmp'] = df_sorted.groupby('race_id').cumcount() + 1
    # Top 3 이내면 1, 아니면 0으로 플래그
    # DataFrame 자체를 리턴하지 않고 Series.map 등으로 매핑하기 편하게 race_id와 hrno 기준 df 반환
    return df_sorted[['race_id', 'pthrHrno', 'rank_tmp']].copy()

def main():
    in_path = os.path.join(config.DATA_MODELING, "modeling_data_ready.csv")
    df = pd.read_csv(in_path, encoding="utf-8-sig", low_memory=False)

    outputs = df[['race_id', 'pthrHrno', 'split_group', config.TARGET_COL]].copy()

    # Rule 1: 레이팅 (pthrRatg) 높을수록 좋음
    r1 = select_top3_by_score(df, 'pthrRatg', ascending=False)
    outputs = outputs.merge(r1, on=['race_id', 'pthrHrno'], how='left')
    outputs['pred_rule1'] = (outputs['rank_tmp'] <= 3).astype(int)
    outputs.drop(columns=['rank_tmp'], inplace=True)
    config.log("Rule 1 (레이팅 상위 3) 생성 완료.")

    # Rule 2: 말 평균순위 (fe_horse_cum_avg_rk) 낮을수록 좋음
    r2 = select_top3_by_score(df, 'fe_horse_cum_avg_rk', ascending=True)
    outputs = outputs.merge(r2, on=['race_id', 'pthrHrno'], how='left')
    outputs['pred_rule2'] = (outputs['rank_tmp'] <= 3).astype(int)
    outputs.drop(columns=['rank_tmp'], inplace=True)
    config.log("Rule 2 (말 누적 평균순위 상위 3) 생성 완료.")

    # Rule 3: 기수 누적 승률 (fe_jcky_cum_win_rate) 높을수록 좋음
    r3 = select_top3_by_score(df, 'fe_jcky_cum_win_rate', ascending=False)
    outputs = outputs.merge(r3, on=['race_id', 'pthrHrno'], how='left')
    outputs['pred_rule3'] = (outputs['rank_tmp'] <= 3).astype(int)
    outputs.drop(columns=['rank_tmp'], inplace=True)
    config.log("Rule 3 (기수 승률 상위 3) 생성 완료.")

    # Rule 4: 조교사 누적 승률 (fe_trar_cum_win_rate) 높을수록 좋음
    r4 = select_top3_by_score(df, 'fe_trar_cum_win_rate', ascending=False)
    outputs = outputs.merge(r4, on=['race_id', 'pthrHrno'], how='left')
    outputs['pred_rule4'] = (outputs['rank_tmp'] <= 3).astype(int)
    outputs.drop(columns=['rank_tmp'], inplace=True)
    config.log("Rule 4 (조교사 승률 상위 3) 생성 완료.")

    # Rule 5: 복합 룰
    # 각 스코어를 동일 기준으로 변환 (백분위 사용)
    # 랭크를 사용하면 범위가 다르므로 pct로 환산
    df['r_score_ratg'] = df.groupby('race_id')['pthrRatg'].rank(pct=True, ascending=True) # 높은게 1(최고점)
    df['r_score_hrk']  = df.groupby('race_id')['fe_horse_cum_avg_rk'].rank(pct=True, ascending=False) # 낮은게 1(최고점)
    df['r_score_jwin'] = df.groupby('race_id')['fe_jcky_cum_win_rate'].rank(pct=True, ascending=True) # 높은게 1
    df['r_score_twin'] = df.groupby('race_id')['fe_trar_cum_win_rate'].rank(pct=True, ascending=True) # 높은게 1
    df['r_penalty_wgt']= df.groupby('race_id')['pthrBurdWgt'].rank(pct=True, ascending=True) # 무거운게 페널티 큼

    df['rule5_score'] = (
        df['r_score_ratg'].fillna(0.5) * 1.5 + 
        df['r_score_hrk'].fillna(0.5) * 2.0 + 
        df['r_score_jwin'].fillna(0.5) * 1.0 + 
        df['r_score_twin'].fillna(0.5) * 1.0 - 
        (df['r_penalty_wgt'].fillna(0.5) * 0.5)
    )

    r5 = select_top3_by_score(df, 'rule5_score', ascending=False)
    outputs = outputs.merge(r5, on=['race_id', 'pthrHrno'], how='left')
    outputs['pred_rule5'] = (outputs['rank_tmp'] <= 3).astype(int)
    outputs.drop(columns=['rank_tmp'], inplace=True)
    config.log("Rule 5 (복합 규칙 상위 3) 생성 완료.")

    # 저장
    out_path = os.path.join(config.DATA_PREDICTIONS, "baseline_rule_predictions.csv")
    outputs.to_csv(out_path, index=False, encoding="utf-8-sig")
    config.log(f"단순 규칙 기반 라벨(1=Top3, 0=그외) 저장 완료: {out_path}")

if __name__ == "__main__":
    main()
