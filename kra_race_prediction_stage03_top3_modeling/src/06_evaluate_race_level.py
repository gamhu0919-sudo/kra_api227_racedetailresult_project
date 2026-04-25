"""
06_evaluate_race_level.py
──────────────────────────────
각 경주별 Test 데이터의 평가를 수행하고, 비교 대상(룰 엔진 vs LightGBM)의 
스코어 테이블을 도출 및 시각화합니다.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

plt = config.setup_plot()

config.log_step(6, "경주 단위 평가 (Race Level Evaluation)")

def compute_ndcg(pred_rank_list, actual_top3_list):
    """
    간이 NDCG@3 계산.
    pred_rank_list: 예측된 1,2,3위 말의 리스트 (순서대로)
    actual_top3_list: 실제 Top3 안에 포함되는 말들
    """
    dcg = 0.0
    for i, horse in enumerate(pred_rank_list):
        if horse in actual_top3_list:
            dcg += 1.0 / np.log2((i+1) + 1)
    
    # IDCG는 실제 Top3 개수만큼 (최대 3개) 앞에서부터 맞췄다고 가정
    idcg = 0.0
    n_ideal = min(len(actual_top3_list), 3)
    for i in range(n_ideal):
        idcg += 1.0 / np.log2((i+1) + 1)
        
    if idcg == 0:
        return 0.0
    return dcg / idcg

def evaluate_predictions(df, true_target_col, pred_target_col, pred_rank_col=None):
    """
    df: 특정 방법론에 의해 예측된 값이 포함된 데이터프레임 (test split 한정 권장)
    pred_target_col: 예측된 Top3 여부 (1 or 0)
    pred_rank_col: NDCG 계산을 위한 정렬 순위. 제공되지 않는 경우 NDCG=N/A
    
    리턴: {precision_at_3, recall_at_3, hit_at_3, avg_correct, ndcg_at_3, test_races}
    """
    metrics = []
    
    for race_id, grp in df.groupby('race_id'):
        # 경주 내 실제 Top3 집합
        actual_top3 = set(grp[grp[true_target_col] == 1]['pthrHrno'])
        n_actual = len(actual_top3)
        if n_actual == 0:
            continue
            
        # 실제 우승마 (여기서는 타겟이 Top3이므로 1위 정보를 정확히 알 수 없음, 
        # 원본 데이터와 머지해야하지만, 모델링 데이터셋에서 target_rank 를 배제했음.
        # 부득이하게 실제 Top3 안에 든 말을 맞혔는지를 넓은 의미의 Hit@3로 정의하거나
        # 원본에서 우승마를 가져올 수 있습니다. 누수 방지를 위해 target_rank를
        # 보존해 두지 않았다면 우승마 특정은 불가능하여 Top3 Hit으로 대체합니다.)
        # 단, 이 단계에서는 구현의 편의를 위해 "실제 Top3를 최소 1개 이상 포함"을 Hit로 근사
        # 완벽한 1위 Hit를 위해서는 원본 CSV 조인이 필요.
        
        # 예측 Top3 집합
        pred_top3 = set(grp[grp[pred_target_col] == 1]['pthrHrno'])
        n_pred = len(pred_top3)
        if n_pred == 0: 
            continue
            
        intersection = actual_top3.intersection(pred_top3)
        n_correct = len(intersection)
        
        precision = n_correct / 3.0 # 항시 3마리 예측이라 가정
        recall = n_correct / n_actual
        hit = 1 if n_correct > 0 else 0 # 1위를 모르는 상태에서의 차선책
        
        ndcg = 0.0
        if pred_rank_col and pred_rank_col in grp.columns:
            # 1,2,3위 순위 리스트
            pred_list = grp[grp[pred_target_col] == 1].sort_values(pred_rank_col)['pthrHrno'].tolist()
            ndcg = compute_ndcg(pred_list, actual_top3)
            
        metrics.append({
            'precision': precision,
            'recall': recall,
            'hit': hit,
            'correct': n_correct,
            'ndcg': ndcg
        })
        
    m_df = pd.DataFrame(metrics)
    if m_df.empty:
        return {}
        
    return {
        'test_race_count': len(m_df),
        'precision_at_3': round(m_df['precision'].mean() * 100, 2),
        'recall_at_3': round(m_df['recall'].mean() * 100, 2),
        'hit_at_3': round(m_df['hit'].mean() * 100, 2),
        'avg_correct_top3_count': round(m_df['correct'].mean(), 3),
        'ndcg_at_3': round(m_df['ndcg'].mean(), 4)
    }

def main():
    # 원본 파일에서 target_rank 불러오기 (Hit@3 정확한 우승마 판별 목적)
    orig_path = os.path.join(config.PREV_STAGE, "data", "processed", "race_data_preprocessed.csv")
    orig_df = pd.read_csv(orig_path, encoding='utf-8-sig', usecols=['race_id', 'pthrHrno', 'target_rank', 'target_is_top3'])

    # 예측 파일 로드
    base_path = os.path.join(config.DATA_PREDICTIONS, "baseline_rule_predictions.csv")
    lgb_path = os.path.join(config.DATA_PREDICTIONS, "lightgbm_top3_predictions.csv")
    
    base_df = pd.read_csv(base_path, encoding="utf-8-sig")
    lgb_df = pd.read_csv(lgb_path, encoding="utf-8-sig")

    # Test Set 필터링
    base_test = base_df[base_df['split_group'] == 'test'].copy()
    lgb_test = lgb_df[lgb_df['split_group'] == 'test'].copy()

    # 타겟 연결 (우승마 확인용 Hit@3 로직 갱신 시 필요 시 활용, 현재는 target_is_top3로 진행)
    # 우승마 Hit@3 재정의
    def hit_1st_calc(grp, pred_col):
        winner = grp[grp['target_rank']==1]['pthrHrno'].values
        if len(winner) == 0: return 0
        preds = grp[grp[pred_col]==1]['pthrHrno'].values
        return 1 if winner[0] in preds else 0

    results = []

    # 1. Random Baseline (근사)
    avg_entries = 10.6 # Stage 02 보고서 참조
    rand_prec = (3 / avg_entries) * 100
    rand_recall= (3 / avg_entries) * 100 # 평균 Top3가 약 2.8~3.0이라고 가정
    rand_act = 3 * (3 / avg_entries)
    
    results.append({
        'method': '1. 무작위 기준선 (Random)',
        'test_race_count': base_test['race_id'].nunique(),
        'precision_at_3': round(rand_prec, 2),
        'recall_at_3': round(rand_recall, 2),
        'hit_at_3': round((3 / avg_entries)*100, 2),
        'avg_correct_top3_count': round(rand_act, 2),
        'ndcg_at_3': 0.0,
        'comment': '산술적 기준값'
    })

    # 원본 합치기 (우승마 식별)
    base_test = base_test.merge(orig_df[['race_id','pthrHrno','target_rank']], on=['race_id','pthrHrno'], how='left')
    lgb_test = lgb_test.merge(orig_df[['race_id','pthrHrno','target_rank']], on=['race_id','pthrHrno'], how='left')

    rules = [
        ('2. 레이팅 상위 3 (Rule 1)', 'pred_rule1'),
        ('3. 말 평균순위 상위 3 (Rule 2)', 'pred_rule2'),
        ('4. 기수 승률 상위 3 (Rule 3)', 'pred_rule3'),
        ('5. 조교사 승률 상위 3 (Rule 4)', 'pred_rule4'),
        ('6. 복합 규칙 점수 (Rule 5)', 'pred_rule5')
    ]

    for name, col in rules:
        ev = evaluate_predictions(base_test, config.TARGET_COL, col)
        
        # Real Hit@3
        hits = base_test.groupby('race_id').apply(lambda g: hit_1st_calc(g, col)).mean() * 100
        ev['hit_at_3'] = round(hits, 2)
        
        ev['method'] = name
        ev['comment'] = '단순 규칙 연산'
        results.append(ev)

    # LightGBM
    ev_lgb = evaluate_predictions(lgb_test, config.TARGET_COL, 'pred_is_top3', 'pred_rank_in_race')
    hits_lgb = lgb_test.groupby('race_id').apply(lambda g: hit_1st_calc(g, 'pred_is_top3')).mean() * 100
    ev_lgb['hit_at_3'] = round(hits_lgb, 2)
    ev_lgb['method'] = '7. LightGBM Top3 모델'
    ev_lgb['comment'] = 'Binary Classifier + 경주 내 정렬'
    results.append(ev_lgb)

    # 결과 테이블 저장
    res_df = pd.DataFrame(results)
    
    col_order = [
        'method', 'test_race_count', 'precision_at_3', 'recall_at_3', 
        'hit_at_3', 'avg_correct_top3_count', 'ndcg_at_3', 'comment'
    ]
    res_df = res_df[col_order]
    
    tbl_path = os.path.join(config.OUT_TABLES, "model_comparison_table.csv")
    res_df.to_csv(tbl_path, index=False, encoding="utf-8-sig")
    config.log(f"모델 비교 결과표 저장 완료: {tbl_path}")

    # 요약 테이블 (metrics 저장)
    res_df.to_csv(os.path.join(config.OUT_METRICS, "model_performance_summary.csv"), index=False, encoding="utf-8-sig")

    # ──────────────────────────────────────────────
    # 시각화 
    # ──────────────────────────────────────────────
    methods_clean = [m.split(". ")[1] if ". " in m else m for m in res_df['method']]
    
    # 1. Precision@3 비교
    fig, ax = plt.subplots(figsize=(10,6))
    bars = ax.barh(methods_clean, res_df['precision_at_3'], color='#5B8DB8')
    ax.set_title('모델별 Precision@3 비교 (Test Set)', fontweight='bold')
    ax.set_xlabel('Precision@3 (%)')
    ax.invert_yaxis()  # 항목 위에서부터 나열
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{bar.get_width()}%", 
                va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUT_FIGS, "model_precision_at3_comparison.png"))
    plt.close()

    # 2. Hit@3 비교
    fig, ax = plt.subplots(figsize=(10,6))
    bars = ax.barh(methods_clean, res_df['hit_at_3'], color='#E07B54')
    ax.set_title('모델별 Hit@3 (1위마 Top3 포함 비율) 비교 (Test Set)', fontweight='bold')
    ax.set_xlabel('Hit@3 (%)')
    ax.invert_yaxis()
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{bar.get_width()}%", 
                va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUT_FIGS, "model_hit_at3_comparison.png"))
    plt.close()

    # 3. 추가: 분포 그래프 (Predicted Probability Distribution)
    fig, ax = plt.subplots(figsize=(8,5))
    lgb_test_pos = lgb_test[lgb_test[config.TARGET_COL] == 1]['pred_top3_prob']
    lgb_test_neg = lgb_test[lgb_test[config.TARGET_COL] == 0]['pred_top3_prob']
    
    ax.hist(lgb_test_neg, bins=50, alpha=0.5, color='gray', label='Actual Not Top3', density=True)
    ax.hist(lgb_test_pos, bins=50, alpha=0.5, color='red', label='Actual Top3', density=True)
    ax.set_title('LightGBM 예측 확률 분포 (실제 Top3 여부별)', fontweight='bold')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUT_FIGS, "predicted_probability_distribution.png"))
    plt.close()

    config.log("비교 시각화 차트 3종 생성 완료 (Precision, Hit, Probability Dist).")

if __name__ == "__main__":
    main()
