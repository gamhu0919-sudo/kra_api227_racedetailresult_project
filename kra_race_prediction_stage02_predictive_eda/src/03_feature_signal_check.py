"""
03_feature_signal_check.py
───────────────────────────
경주 내 상대 피처 생성 + 상대 피처의 예측 신호 검증

생성되는 상대 피처:
  - rating_rank_in_race
  - rating_pct_rank_in_race
  - rating_zscore_in_race
  - weight_rank_in_race
  - weight_zscore_in_race
  - jockey_winrate_rank_in_race
  - jockey_top3_rate_rank_in_race
  - trainer_winrate_rank_in_race
  - horse_avg_rank_rank_in_race   (낮을수록 좋음 → 오름차순 순위)
  - horse_winrate_rank_in_race
  - horse_experience_rank_in_race

산출물:
  data/processed/race_relative_features.csv
  outputs/tables/relative_feature_lift_analysis.csv
  outputs/figures/relative_feature_lift_*.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from config_loader import (
    DATA_PROCESSED, OUT_TABLES, OUT_FIGS,
    LEAKAGE_COLS, MIN_SAMPLE_N, log, log_step, setup_plot
)

plt = setup_plot()

# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
processed_path = os.path.join(DATA_PROCESSED, "race_data_preprocessed.csv")
df = pd.read_csv(processed_path, encoding="utf-8-sig", low_memory=False)
df["schdRaceDt"] = pd.to_datetime(df["schdRaceDt"], errors="coerce")

df_v = df[df["is_valid_race"] == True].copy()
BASELINE = df_v["target_is_top3"].mean()
log(f"기준선: {BASELINE*100:.4f}%")
log(f"유효 데이터: {len(df_v):,}행 / {df_v['race_id'].nunique():,}경주")

# ─────────────────────────────────────────────
# Step 1: 경주 내 상대 피처 생성
# ─────────────────────────────────────────────
log_step(1, "경주 내 상대 피처 생성")

def rank_within(series, ascending=True):
    """경주 내 순위 (ascending=True → 낮은 값이 1위)"""
    return series.rank(method="min", ascending=ascending)

def pct_rank_within(series, ascending=True):
    """경주 내 백분위 순위 (0~1)"""
    return series.rank(method="average", ascending=ascending, pct=True)

def zscore_within(series):
    """경주 내 Z-점수"""
    m, s = series.mean(), series.std()
    if pd.isna(s) or s == 0:
        return pd.Series(0.0, index=series.index)
    return (series - m) / s


REL_FEAT_DEFS = [
    # (새 컬럼명, 원본 컬럼, ascending, 유형)
    # ascending=False → 높은 값이 1위(순위 1=최강)
    # ascending=True  → 낮은 값이 1위(fe_horse_cum_avg_rk: 낮을수록 좋음)
    ("rating_rank_in_race",           "pthrRatg",              False, "rank"),
    ("rating_pct_rank_in_race",       "pthrRatg",              False, "pct"),
    ("rating_zscore_in_race",         "pthrRatg",              None,  "zscore"),
    ("weight_rank_in_race",           "pthrBurdWgt",           False, "rank"),
    ("weight_zscore_in_race",         "pthrBurdWgt",           None,  "zscore"),
    ("jockey_winrate_rank_in_race",   "fe_jcky_cum_win_rate",  False, "rank"),
    ("jockey_top3_rate_rank_in_race", "fe_jcky_cum_top3_rate", False, "rank"),
    ("trainer_winrate_rank_in_race",  "fe_trar_cum_win_rate",  False, "rank"),
    # fe_horse_cum_avg_rk: 낮을수록 좋음 → ascending=True(낮은 값이 1위)
    ("horse_avg_rank_rank_in_race",   "fe_horse_cum_avg_rk",   True,  "rank"),
    ("horse_winrate_rank_in_race",    "fe_horse_cum_win_rate",  False, "rank"),
    ("horse_experience_rank_in_race", "fe_horse_race_count",    False, "rank"),
]

for new_col, src_col, ascending, feat_type in REL_FEAT_DEFS:
    if src_col not in df_v.columns:
        log(f"  [{src_col}] 컬럼 없음 — 스킵")
        continue

    if feat_type == "rank":
        df_v[new_col] = df_v.groupby("race_id")[src_col].transform(
            lambda x: rank_within(x, ascending=ascending)
        )
    elif feat_type == "pct":
        df_v[new_col] = df_v.groupby("race_id")[src_col].transform(
            lambda x: pct_rank_within(x, ascending=ascending)
        )
    elif feat_type == "zscore":
        df_v[new_col] = df_v.groupby("race_id")[src_col].transform(zscore_within)

    log(f"  {new_col} 생성 완료")

# 상대 피처 컬럼 목록
REL_FEAT_COLS = [f[0] for f in REL_FEAT_DEFS if f[0] in df_v.columns]

# 저장
save_cols = (
    ["race_id", "schdRaceDt", "pthrHrno", "pthrHrnm",
     "target_rank", "target_is_top3"] + REL_FEAT_COLS
)
save_cols = [c for c in save_cols if c in df_v.columns]

rel_feat_path = os.path.join(DATA_PROCESSED, "race_relative_features.csv")
df_v[save_cols].to_csv(rel_feat_path, index=False, encoding="utf-8-sig")
log(f"  race_relative_features.csv 저장: {len(df_v):,}행")


# ─────────────────────────────────────────────
# Step 2: 상대 피처 예측 신호 검증
# ─────────────────────────────────────────────
log_step(2, "상대 피처 예측 신호 검증 (경주 내 순위별 Top3 비율)")

lift_rows = []

for new_col, _, _, feat_type in REL_FEAT_DEFS:
    if new_col not in df_v.columns:
        continue

    if feat_type in ["rank", "pct"]:
        # 경주 내 1위, 2위, 3위, 4~6위, 나머지로 구분
        def categorize_rank(r, total_range=None):
            try:
                r = int(r)
            except Exception:
                return "기타"
            if r == 1:
                return "경주내_1위"
            elif r == 2:
                return "경주내_2위"
            elif r == 3:
                return "경주내_3위"
            elif r <= 6:
                return "경주내_4~6위"
            else:
                return "경주내_7위이하"

        if feat_type == "rank":
            df_v["__grp"] = df_v[new_col].apply(categorize_rank)
        else:  # pct
            df_v["__grp"] = pd.cut(
                df_v[new_col],
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.01],
                labels=["상위0~20%", "상위20~40%", "중간40~60%", "하위20~40%", "하위0~20%"],
                include_lowest=True
            )

    elif feat_type == "zscore":
        df_v["__grp"] = pd.cut(
            df_v[new_col],
            bins=[-99, -1.5, -0.5, 0.5, 1.5, 99],
            labels=["Z<-1.5", "-1.5<=Z<-0.5", "-0.5<=Z<0.5", "0.5<=Z<1.5", "Z>=1.5"]
        )

    # 그룹별 Lift
    grp_stats = df_v.groupby("__grp", observed=True).agg(
        n_entries  = ("target_is_top3", "count"),
        n_top3     = ("target_is_top3", "sum"),
    ).reset_index()
    grp_stats["top3_rate"]      = grp_stats["n_top3"] / grp_stats["n_entries"]
    grp_stats["diff_from_base"] = grp_stats["top3_rate"] - BASELINE
    grp_stats["lift"]           = grp_stats["top3_rate"] / BASELINE
    grp_stats["rel_feature"]    = new_col
    grp_stats["note"]           = grp_stats["n_entries"].apply(
        lambda x: "참고용(n<100)" if x < MIN_SAMPLE_N else ""
    )
    grp_stats = grp_stats.rename(columns={"__grp": "group"})
    lift_rows.append(grp_stats)

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#E07B54" if l >= 1.2 else "#5B8DB8" if l >= 1.0 else "#BBBBBB"
              for l in grp_stats["lift"]]
    bars = ax.bar([str(g) for g in grp_stats["group"]], grp_stats["top3_rate"] * 100,
                  color=colors, edgecolor="white")
    ax.axhline(BASELINE * 100, color="red", linestyle="--", linewidth=1.5,
               label=f"기준선 {BASELINE*100:.1f}%")
    for bar, (_, row) in zip(bars, grp_stats.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"x{row['lift']:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{new_col} — 경주 내 위치별 Top3 비율", fontsize=12, fontweight="bold")
    ax.set_ylabel("Top3 비율 (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGS, f"relative_feature_lift_{new_col}.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    df_v.drop(columns=["__grp"], errors="ignore", inplace=True)
    log(f"  [{new_col}] 최고 Lift: {grp_stats['lift'].max():.3f}")

# 통합 저장
rel_lift_df = pd.concat(lift_rows, ignore_index=True)
col_order = ["rel_feature", "group", "n_entries", "n_top3", "top3_rate",
             "diff_from_base", "lift", "note"]
rel_lift_df = rel_lift_df[[c for c in col_order if c in rel_lift_df.columns]]
rel_lift_df.to_csv(
    os.path.join(OUT_TABLES, "relative_feature_lift_analysis.csv"),
    index=False, encoding="utf-8-sig"
)
log(f"  relative_feature_lift_analysis.csv 저장 완료 ({len(rel_lift_df)}행)")


# ─────────────────────────────────────────────
# Step 3: 절대 vs 상대 피처 비교
# ─────────────────────────────────────────────
log_step(3, "절대 피처 vs 경주 내 상대 피처 비교")

compare_pairs = [
    ("pthrRatg",            "rating_rank_in_race",        "레이팅: 절대값 vs 경주 내 순위"),
    ("fe_jcky_cum_win_rate","jockey_winrate_rank_in_race", "기수 승률: 절대값 vs 경주 내 순위"),
    ("fe_horse_cum_avg_rk", "horse_avg_rank_rank_in_race", "말 평균순위: 절대값 vs 경주 내 순위"),
]

compare_results = []
for abs_col, rel_col, label in compare_pairs:
    if abs_col not in df_v.columns or rel_col not in df_v.columns:
        continue

    # 절대 피처: 상위 33%
    if abs_col == "fe_horse_cum_avg_rk":
        abs_top = df_v[df_v[abs_col] <= df_v[abs_col].quantile(0.33)]
    else:
        abs_top = df_v[df_v[abs_col] >= df_v[abs_col].quantile(0.67)]

    # 상대 피처: 경주 내 1위
    rel_top = df_v[df_v[rel_col] == 1]

    abs_rate = abs_top["target_is_top3"].mean() if len(abs_top) > 0 else 0
    rel_rate = rel_top["target_is_top3"].mean() if len(rel_top) > 0 else 0

    compare_results.append({
        "비교 쌍": label,
        "절대 피처": abs_col,
        "절대 피처 상위33% Top3비율": round(abs_rate, 4),
        "절대 피처 Lift": round(abs_rate / BASELINE, 3),
        "상대 피처": rel_col,
        "상대 피처 경주내1위 Top3비율": round(rel_rate, 4),
        "상대 피처 Lift": round(rel_rate / BASELINE, 3),
        "상대피처 우월 여부": "상대피처 우월" if rel_rate > abs_rate else "절대피처 우월",
    })
    log(f"  [{label}] 절대:{abs_rate*100:.1f}% vs 상대:{rel_rate*100:.1f}%")

compare_df = pd.DataFrame(compare_results)
compare_df.to_csv(
    os.path.join(OUT_TABLES, "absolute_vs_relative_comparison.csv"),
    index=False, encoding="utf-8-sig"
)


print("\n[완료] 03_feature_signal_check.py 실행 완료")
print(f"  - race_relative_features.csv: {len(REL_FEAT_COLS)}개 상대 피처")
print(f"  - relative_feature_lift_analysis.csv")
print(f"  - absolute_vs_relative_comparison.csv")
