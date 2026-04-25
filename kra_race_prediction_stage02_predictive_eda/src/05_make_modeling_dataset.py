"""
05_make_modeling_dataset.py
────────────────────────────────────────
단순 규칙 베이스라인 성능 측정 + 모델링용 데이터셋 최종 생성

포함 내용:
  1. 단순 규칙 기반 베이스라인 성능 (경주 단위 평가)
     - 레이팅 상위 3
     - 말 누적 평균순위 상위 3
     - 기수 승률 상위 3
     - 조교사 승률 상위 3
     - 복합 점수 상위 3 (rule_score)
     
  2. 모델링 데이터셋 생성
     - 누수 컬럼 제거
     - 상대 피처 합류
     - is_valid_race 플래그
     - train/test 분할 그룹 후보

산출물:
  outputs/tables/baseline_rule_performance.csv
  data/modeling_ready/modeling_dataset_top3.csv
  data/modeling_ready/modeling_dataset_rank.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from config_loader import (
    DATA_PROCESSED, DATA_MODELING, OUT_TABLES, REPORTS,
    LEAKAGE_COLS, ID_COLS, TARGET_COLS, log, log_step
)

# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
processed_path = os.path.join(DATA_PROCESSED, "race_data_preprocessed.csv")
rel_feat_path  = os.path.join(DATA_PROCESSED, "race_relative_features.csv")

df = pd.read_csv(processed_path, encoding="utf-8-sig", low_memory=False)
df["schdRaceDt"] = pd.to_datetime(df["schdRaceDt"], errors="coerce")

# 상대 피처 합류
rel_df = pd.read_csv(rel_feat_path, encoding="utf-8-sig", low_memory=False)
rel_cols = [c for c in rel_df.columns if c not in ["schdRaceDt", "pthrHrnm",
                                                     "target_rank", "target_is_top3"]]
df = df.merge(rel_df[rel_cols], on=["race_id", "pthrHrno"], how="left")

# 유효 경주만 사용 (베이스라인 + 모델링)
df_v = df[df["is_valid_race"] == True].copy()
BASELINE = df_v["target_is_top3"].mean()

log(f"데이터 로드: {len(df_v):,}행 / {df_v['race_id'].nunique():,}경주")
log(f"전체 기준선: {BASELINE*100:.4f}%")


# ─────────────────────────────────────────────
# Step 1: 단순 규칙 베이스라인 성능 측정
# ─────────────────────────────────────────────
log_step(1, "단순 규칙 베이스라인 성능 측정 (경주 단위)")

def evaluate_rule(df_in, score_col, rule_name, higher_is_better=True, top_n=3):
    """
    각 경주에서 score_col 기준 상위 top_n마리를 Top3로 예측 후 경주 단위 평가

    평가 지표:
    - Race Hit@3: 예측 Top3 중 실제 Top3에 들어간 말 수 (경주 평균)
    - Winner_in_Top3: 실제 1위가 예측 Top3 안에 있는 경주 비율
    - Precision@3: 예측 3마리 중 실제 Top3인 비율 (경주 평균)
    - Avg_NDCG@3: 순위 가중 정확도 (간이)
    """
    race_results = []

    for race_id, grp in df_in.groupby("race_id"):
        n = len(grp)
        if n < 3:
            continue

        # top_n 선정
        if higher_is_better:
            pred_top = grp.nlargest(top_n, score_col, keep="first")["pthrHrno"].values
        else:
            pred_top = grp.nsmallest(top_n, score_col, keep="first")["pthrHrno"].values

        actual_top3 = grp[grp["target_is_top3"] == 1]["pthrHrno"].values
        actual_winner = grp[grp["target_rank"] == 1]["pthrHrno"].values

        # 지표 계산
        n_hit         = len(set(pred_top) & set(actual_top3))
        winner_in_top = int(len(set(pred_top) & set(actual_winner)) > 0)
        precision     = n_hit / top_n

        # NDCG@3 (간이): 예측 순서대로 실제 top3이면 gain
        dcg = 0.0
        for rank_i, horse in enumerate(pred_top, 1):
            if horse in actual_top3:
                dcg += 1.0 / np.log2(rank_i + 1)
        ideal_dcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(actual_top3), top_n) + 1))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

        race_results.append({
            "race_id":     race_id,
            "n_entries":   n,
            "n_hit":       n_hit,
            "winner_hit":  winner_in_top,
            "precision":   precision,
            "ndcg":        ndcg,
        })

    results_df = pd.DataFrame(race_results)
    n_races = len(results_df)
    if n_races == 0:
        return {}

    summary = {
        "rule_name":           rule_name,
        "n_races_evaluated":   n_races,
        "baseline_top3_rate":  round(BASELINE * 100, 2),
        "race_hit_at3_avg":    round(results_df["n_hit"].mean(), 4),
        "winner_in_top3_pct":  round(results_df["winner_hit"].mean() * 100, 2),
        "precision_at3_avg":   round(results_df["precision"].mean() * 100, 2),
        "ndcg_at3_avg":        round(results_df["ndcg"].mean(), 4),
        "random_baseline_hit": round(3 / (df_in.groupby("race_id").size().mean()) * top_n, 4),
    }
    return summary


# 복합 점수 생성
# 각 피처를 경주 내 백분위로 변환 후 합산 (방향 보정 포함)
def make_rule_score(df_in):
    score = pd.Series(0.0, index=df_in.index)
    scoreable = {
        "pthrRatg":             True,   # 높을수록 좋음
        "fe_jcky_cum_win_rate": True,
        "fe_horse_cum_avg_rk":  False,  # 낮을수록 좋음
        "fe_trar_cum_win_rate": True,
        "fe_horse_cum_win_rate":True,
    }
    added = 0
    for col, higher_good in scoreable.items():
        if col not in df_in.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df_in[col]):
            continue
        pct = df_in.groupby("race_id")[col].transform(
            lambda x: x.rank(pct=True)
        )
        if not higher_good:
            pct = 1 - pct
        score += pct.fillna(0.5)
        added += 1
    return score / max(added, 1)

df_v["rule_score"] = make_rule_score(df_v)

# 5가지 규칙 평가
RULES = [
    ("pthrRatg",             "레이팅 상위3",           True),
    ("fe_horse_cum_avg_rk",  "말평균순위 상위3",        False),
    ("fe_jcky_cum_win_rate", "기수승률 상위3",          True),
    ("fe_trar_cum_win_rate", "조교사승률 상위3",        True),
    ("rule_score",           "복합점수(rule_score) 상위3", True),
]

perf_rows = []
for col, name, higher in RULES:
    if col not in df_v.columns:
        log(f"  [{col}] 없음 스킵")
        continue
    result = evaluate_rule(df_v, col, name, higher_is_better=higher)
    if result:
        perf_rows.append(result)
        log(f"  [{name}]")
        log(f"    Precision@3: {result['precision_at3_avg']}%")
        log(f"    Winner-in-Top3: {result['winner_in_top3_pct']}%")
        log(f"    NDCG@3: {result['ndcg_at3_avg']:.4f}")

perf_df = pd.DataFrame(perf_rows)

# 기준선 대비 향상 계산
if len(perf_df) > 0:
    avg_top3_pct = BASELINE * 100
    perf_df["vs_random_winner_pct"] = round(
        perf_df["winner_in_top3_pct"] / (avg_top3_pct) - 1, 4
    ) * 100

perf_df.to_csv(
    os.path.join(OUT_TABLES, "baseline_rule_performance.csv"),
    index=False, encoding="utf-8-sig"
)
log(f"  baseline_rule_performance.csv 저장 완료 ({len(perf_df)}개 규칙)")


# ─────────────────────────────────────────────
# Step 2: 모델링용 데이터셋 생성
# ─────────────────────────────────────────────
log_step(2, "모델링용 데이터셋 생성")

# 제거할 컬럼: 누수 + 타깃 원본(rsutRk)
DROP_COLS = [c for c in LEAKAGE_COLS if c in df_v.columns]

# 피처 컬럼 목록 정의
FEATURE_COLS = [
    # 말 관련 (경주 전 정보)
    "pthrAg", "pthrBurdWgt", "pthrGtno", "pthrNtnlty", "pthrRatg",
    # 파생 피처 (경주 전)
    "fe_horse_weight", "fe_weight_diff", "fe_ratg_per_weight",
    "fe_horse_cum_win_rate", "fe_horse_cum_avg_rk", "fe_horse_race_count",
    "fe_jcky_cum_win_rate", "fe_jcky_cum_top3_rate",
    "fe_trar_cum_win_rate",
    "fe_race_dist", "fe_track_humidity", "fe_month", "fe_season",
    # 경주 조건
    "cndRaceClas", "cndBurdGb", "cndRatg", "cndAg", "cndGndr",
    # 환경
    "rsutTrckStus", "rsutWetr",
    # 기수 수당
    "hrmJckyAlw",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in df_v.columns]

# 상대 피처 컬럼
REL_FEAT_COLS = [c for c in df_v.columns if c.endswith("_in_race") or c.endswith("_zscore_in_race")]

# 식별자 컬럼
KEEP_ID_COLS = [
    "race_id", "schdRaceDt", "schdRaceNo",
    "pthrHrno", "pthrHrnm",
    "hrmJckyId", "hrmJckyNm",
    "hrmTrarId", "hrmTrarNm",
]
KEEP_ID_COLS = [c for c in KEEP_ID_COLS if c in df_v.columns]

# train/test 후보 그룹 (time-based)
CUTOFF_DATE = pd.Timestamp("2025-07-01")
df_v["train_test_split_group_candidate"] = df_v["schdRaceDt"].apply(
    lambda d: "train" if pd.notna(d) and d < CUTOFF_DATE else "test"
)
log(f"  Train/Test 분할: {(df_v['train_test_split_group_candidate']=='train').sum():,} / {(df_v['train_test_split_group_candidate']=='test').sum():,}")

# 누수 체크 플래그
df_v["is_leakage_free"] = True  # 누수 컬럼을 제거했으므로 True

# ── Top3 이진 분류용 데이터셋 ──
top3_cols = (
    KEEP_ID_COLS
    + FEATURE_COLS
    + REL_FEAT_COLS
    + ["target_is_top3", "target_rank",
       "is_valid_race", "is_leakage_free", "train_test_split_group_candidate"]
)
top3_cols = [c for c in dict.fromkeys(top3_cols) if c in df_v.columns]
df_top3 = df_v[top3_cols].copy()
df_top3.to_csv(
    os.path.join(DATA_MODELING, "modeling_dataset_top3.csv"),
    index=False, encoding="utf-8-sig"
)
log(f"  modeling_dataset_top3.csv: {df_top3.shape} ({df_top3.columns.nunique()}개 컬럼)")

# ── 순위 예측용 데이터셋 ──
rank_cols = (
    KEEP_ID_COLS
    + FEATURE_COLS
    + REL_FEAT_COLS
    + ["target_rank", "target_is_top3",
       "is_valid_race", "is_leakage_free", "train_test_split_group_candidate"]
)
rank_cols = [c for c in dict.fromkeys(rank_cols) if c in df_v.columns]
df_rank = df_v[rank_cols].copy()
df_rank.to_csv(
    os.path.join(DATA_MODELING, "modeling_dataset_rank.csv"),
    index=False, encoding="utf-8-sig"
)
log(f"  modeling_dataset_rank.csv: {df_rank.shape}")


# ─────────────────────────────────────────────
# Step 3: 모델링 데이터셋 요약 통계
# ─────────────────────────────────────────────
log_step(3, "모델링 데이터셋 피처 요약 저장")

feat_summary_rows = []
for col in FEATURE_COLS + REL_FEAT_COLS:
    if col not in df_top3.columns:
        continue
    s = df_top3[col]
    is_num = pd.api.types.is_numeric_dtype(s)
    feat_summary_rows.append({
        "feature":       col,
        "is_numeric":    is_num,
        "missing_pct":   round(s.isna().sum() / len(s) * 100, 2),
        "unique_count":  s.nunique(),
        "mean":          round(s.mean(), 4) if is_num else None,
        "std":           round(s.std(), 4) if is_num else None,
        "min":           round(s.min(), 4) if is_num else None,
        "max":           round(s.max(), 4) if is_num else None,
        "category":      "상대피처" if col in REL_FEAT_COLS else "절대피처",
    })

feat_summary_df = pd.DataFrame(feat_summary_rows)
feat_summary_df.to_csv(
    os.path.join(OUT_TABLES, "modeling_feature_summary.csv"),
    index=False, encoding="utf-8-sig"
)
log(f"  modeling_feature_summary.csv 저장 ({len(feat_summary_df)}개 피처)")


print("\n[완료] 05_make_modeling_dataset.py 실행 완료")
print(f"  - baseline_rule_performance.csv ({len(perf_df)}개 규칙 평가)")
print(f"  - modeling_dataset_top3.csv: {df_top3.shape}")
print(f"  - modeling_dataset_rank.csv: {df_rank.shape}")
if len(perf_df) > 0:
    best_rule = perf_df.loc[perf_df["precision_at3_avg"].idxmax()]
    print(f"  - 최고 성능 규칙: [{best_rule['rule_name']}] Precision@3={best_rule['precision_at3_avg']}%")
