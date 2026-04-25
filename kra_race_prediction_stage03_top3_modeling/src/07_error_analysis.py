"""
07_error_analysis.py
──────────────────────────────
LightGBM 모델 결과 오분류 분석 스크립트.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

plt = config.setup_plot()
config.log_step(7, "테스트 결과 기반 오분류(Error) 모델 특성 분석")

def hit_target_calc(grp, pred_col, target_col):
    """실제 우승마(1위) Hit을 target_rank를 통해 계산"""
    winner_hrnos = grp[grp[target_col]==1]['pthrHrno'].values
    if len(winner_hrnos) == 0: return 0
    preds = grp[grp[pred_col]==1]['pthrHrno'].values
    return 1 if winner_hrnos[0] in preds else 0

def main():
    # 1. 파일 로드
    lgb_path = os.path.join(config.DATA_PREDICTIONS, "lightgbm_top3_predictions.csv")
    lgb_df = pd.read_csv(lgb_path, encoding="utf-8-sig")

    feat_path = os.path.join(config.DATA_MODELING, "modeling_data_ready.csv")
    feat_df = pd.read_csv(feat_path, encoding="utf-8-sig", low_memory=False)

    orig_path = os.path.join(config.PREV_STAGE, "data", "processed", "race_data_preprocessed.csv")
    orig_df = pd.read_csv(orig_path, encoding='utf-8-sig', usecols=['race_id', 'pthrHrno', 'target_rank', 'target_is_top3'])

    # Test Set만 추리기
    test_lgb = lgb_df[lgb_df['split_group'] == 'test'].copy()
    
    # 2. 메타 및 거리/등급 등 피처 조인
    join_cols = ['race_id', 'pthrHrno', 'fe_race_dist', 'cndRaceClas']
    test_joined = test_lgb.merge(feat_df[join_cols], on=['race_id', 'pthrHrno'], how='left')
    test_joined = test_joined.merge(orig_df[['race_id', 'pthrHrno', 'target_rank']], on=['race_id', 'pthrHrno'], how='left')

    analysis_rows = []
    
    for race_id, grp in test_joined.groupby('race_id'):
        dist = grp['fe_race_dist'].iloc[0]
        r_class = grp['cndRaceClas'].iloc[0]

        actual_top3 = set(grp[grp['target_is_top3'] == 1]['pthrHrno'])
        pred_top3 = set(grp[grp['pred_is_top3'] == 1]['pthrHrno'])
        
        n_correct = len(actual_top3.intersection(pred_top3))
        is_hit = hit_target_calc(grp, 'pred_is_top3', 'target_rank')
        
        analysis_rows.append({
            'race_id': race_id,
            'fe_race_dist': dist,
            'cndRaceClas': r_class,
            'n_correct': n_correct,
            'is_hit': is_hit
        })

    adf = pd.DataFrame(analysis_rows)

    # 3. Good Races / Bad Races 분류
    # Good Races: Top 3 중 2마리 이상 맞췄거나 우승마를 적중함
    good_cond = (adf['n_correct'] >= 2) | (adf['is_hit'] == 1)
    
    # Bad Races: 1마리도 못 맞췄거나, 0개
    bad_cond = (adf['n_correct'] == 0)

    good_races = adf[good_cond].copy()
    bad_races = adf[bad_cond].copy()

    config.log(f"Good Races 수: {len(good_races)} / Bad Races 수: {len(bad_races)}")

    good_races.to_csv(os.path.join(config.OUT_TABLES, "good_prediction_races.csv"), index=False, encoding="utf-8-sig")
    bad_races.to_csv(os.path.join(config.OUT_TABLES, "bad_prediction_races.csv"), index=False, encoding="utf-8-sig")

    # 4. 거리별 / 등급별 집계
    # 거리별
    dist_info = adf.groupby('fe_race_dist').agg(
        total_races=pd.NamedAgg(column='race_id', aggfunc='count'),
        avg_correct=pd.NamedAgg(column='n_correct', aggfunc='mean'),
        hit_ratio=pd.NamedAgg(column='is_hit', aggfunc='mean')
    ).reset_index()
    dist_info['hit_ratio'] = round(dist_info['hit_ratio'] * 100, 2)
    dist_info['avg_correct'] = round(dist_info['avg_correct'], 2)
    
    dist_info.to_csv(os.path.join(config.OUT_TABLES, "error_analysis_by_distance.csv"), index=False, encoding="utf-8-sig")

    # 등급별
    class_info = adf.groupby('cndRaceClas').agg(
        total_races=pd.NamedAgg(column='race_id', aggfunc='count'),
        avg_correct=pd.NamedAgg(column='n_correct', aggfunc='mean'),
        hit_ratio=pd.NamedAgg(column='is_hit', aggfunc='mean')
    ).reset_index()
    class_info['hit_ratio'] = round(class_info['hit_ratio'] * 100, 2)
    class_info['avg_correct'] = round(class_info['avg_correct'], 2)
    
    class_info.to_csv(os.path.join(config.OUT_TABLES, "error_analysis_by_class.csv"), index=False, encoding="utf-8-sig")

    # 5. 시각화 (거리/등급별 Hit@3 성능)
    # 거리별
    # 표본이 10개 이상인 거리만 시각화
    dist_valid = dist_info[dist_info['total_races'] >= 10].sort_values('fe_race_dist')
    if not dist_valid.empty:
        fig, ax = plt.subplots(figsize=(10,5))
        bars = ax.bar([str(x) for x in dist_valid['fe_race_dist']], dist_valid['hit_ratio'], color='#5B8DB8')
        ax.set_title('거리별 Test Hit@3 성능', fontweight='bold')
        ax.set_xlabel('거리 (m)')
        ax.set_ylabel('Hit@3 (%)')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f"{bar.get_height()}%", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUT_FIGS, "performance_by_distance.png"))
        plt.close()

    # 등급별
    class_valid = class_info[class_info['total_races'] >= 10].sort_values('hit_ratio', ascending=False)
    if not class_valid.empty:
        fig, ax = plt.subplots(figsize=(10,5))
        bars = ax.bar(class_valid['cndRaceClas'].astype(str), class_valid['hit_ratio'], color='#E07B54')
        ax.set_title('등급별 Test Hit@3 성능', fontweight='bold')
        ax.set_xlabel('등급')
        ax.set_ylabel('Hit@3 (%)')
        plt.xticks(rotation=45)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f"{bar.get_height()}%", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUT_FIGS, "performance_by_class.png"))
        plt.close()

    config.log("오분류 데이터 검출 및 거리별/등급별 분석 시각화 완료.")

if __name__ == "__main__":
    main()
