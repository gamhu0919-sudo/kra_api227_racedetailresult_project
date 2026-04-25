"""
02_predictive_eda.py
─────────────────────────────────────
예측형 EDA - 기준선 대비 변수별 Lift 분석
핵심: "전체 Top3 비율은 기준선이다"

분석 내용:
  - 변수별 분위 그룹 → 그룹별 Top3 비율 계산
  - Lift = 그룹 Top3 비율 / 전체 기준선
  - 표본 수 100 미만 그룹은 참고용 표시
  - 거리별 / 등급별 세그먼트 분석

산출물:
  outputs/tables/feature_bucket_top3_lift.csv
  outputs/tables/leakage_columns.csv
  outputs/tables/distance_segment_signal.csv
  outputs/tables/class_segment_signal.csv
  outputs/figures/feature_lift_barplot_*.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from config_loader import (
    DATA_PROCESSED, OUT_TABLES, OUT_FIGS, OUT_DIAG,
    LEAKAGE_COLS, MIN_SAMPLE_N, log, log_step, setup_plot
)

plt = setup_plot()

# ─────────────────────────────────────────────
# 데이터 로드 (01에서 생성된 전처리 파일)
# ─────────────────────────────────────────────
processed_path = os.path.join(DATA_PROCESSED, "race_data_preprocessed.csv")
df = pd.read_csv(processed_path, encoding="utf-8-sig", low_memory=False)
df["schdRaceDt"] = pd.to_datetime(df["schdRaceDt"], errors="coerce")

# 유효 경주만 사용
df_v = df[df["is_valid_race"] == True].copy()

# 전체 기준선
BASELINE = df_v["target_is_top3"].mean()
log(f"전체 기준선(baseline_top3_rate): {BASELINE*100:.4f}%")

# ─────────────────────────────────────────────
# Step 1: 누수 컬럼 목록 저장
# ─────────────────────────────────────────────
log_step(1, "누수 컬럼 목록 저장")

LEAKAGE_REASON = {
    "rsutRk":            "타깃 원본 (직접 데이터 누수)",
    "rsutRaceRcd":       "레이스 타임 — 경주 완료 후 기록",
    "rsutMargin":        "마신차 — 경주 완료 후 기록",
    "rsutRkAdmny":       "입상 여부 — 경주 완료 후 확정",
    "rsutRkPurse":       "상금 — 경주 결과에 의존",
    "rsutQnlaPrice":     "연세가격 — 경주 후 산출",
    "rsutWinPrice":      "배당금 — 경주 후 확정 (경주 전 오즈와 다름)",
    "rsutRkRemk":        "결과 비고 — 경주 후 기록",
    "rsutRlStrtTim":     "실제 출발 시각 — 경주 후 기록",
    "rsutStrtTimChgRs":  "출발 시각 변경 사유 — 경주 후 기록",
}

leakage_df = pd.DataFrame([
    {"column": col, "reason": reason, "is_in_data": col in df.columns}
    for col, reason in LEAKAGE_REASON.items()
])
leakage_df.to_csv(os.path.join(OUT_TABLES, "leakage_columns.csv"), index=False, encoding="utf-8-sig")
log(f"  누수 컬럼 {len(leakage_df)}개 저장 완료")

# ─────────────────────────────────────────────
# 공통 함수: 변수 → Lift 계산
# ─────────────────────────────────────────────

def compute_lift(df_in, group_col, baseline=BASELINE, min_n=MIN_SAMPLE_N):
    """
    범주형/분위 컬럼 기준으로 Top3 Lift 계산
    반환: DataFrame with [group, n_entries, n_top3, top3_rate, diff_from_base, lift, note]
    """
    grp = df_in.groupby(group_col, observed=True).agg(
        n_entries  = ("target_is_top3", "count"),
        n_top3     = ("target_is_top3", "sum"),
    ).reset_index()
    grp["top3_rate"]       = grp["n_top3"] / grp["n_entries"]
    grp["diff_from_base"]  = grp["top3_rate"] - baseline
    grp["lift"]            = grp["top3_rate"] / baseline
    grp["note"]            = grp["n_entries"].apply(
        lambda x: "참고용(n<100)" if x < min_n else ""
    )
    grp["feature"]         = group_col
    return grp.sort_values("lift", ascending=False)


def make_quantile_col(series, q=5, prefix="Q"):
    """분위수 구간 컬럼 생성 (중복 처리 포함)"""
    try:
        labels = [f"{prefix}{i+1}" for i in range(q)]
        return pd.qcut(series, q=q, labels=labels, duplicates="drop")
    except Exception:
        try:
            labels = [f"{prefix}{i+1}" for i in range(q-1)]
            return pd.qcut(series, q=q-1, labels=labels, duplicates="drop")
        except Exception:
            return pd.cut(series, bins=3, labels=["Low","Mid","High"])


def plot_lift_bar(lift_df, feature_name, baseline, save_dir):
    """Lift 막대 그래프 저장"""
    fig, ax = plt.subplots(figsize=(10, 5))
    grp_col = lift_df.columns[0]  # 첫 번째 컬럼이 그룹명

    # 표본 수 적은 그룹 구분
    colors = []
    for _, row in lift_df.iterrows():
        if row["note"]:
            colors.append("#BBBBBB")   # 회색(참고용)
        elif row["lift"] >= 1.2:
            colors.append("#E07B54")   # 주황(높은 Lift)
        elif row["lift"] >= 1.0:
            colors.append("#5B8DB8")   # 파랑(기준 이상)
        else:
            colors.append("#7FBFBF")   # 청록(기준 이하)

    bars = ax.bar(
        [str(x) for x in lift_df[grp_col]],
        lift_df["top3_rate"] * 100,
        color=colors, edgecolor="white", linewidth=0.5
    )
    ax.axhline(baseline * 100, color="red", linestyle="--", linewidth=1.5,
               label=f"기준선 {baseline*100:.1f}%")

    # 바 위에 Lift 값 표시
    for bar, (_, row) in zip(bars, lift_df.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"x{row['lift']:.2f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_title(f"{feature_name} — 분위별 Top3 비율 vs 기준선", fontsize=12, fontweight="bold")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Top3 비율 (%)")
    ax.legend()
    plt.tight_layout()

    fname = os.path.join(save_dir, f"feature_lift_barplot_{feature_name}.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    return fname


# ─────────────────────────────────────────────
# Step 2: 변수별 분위 분석 (연속형)
# ─────────────────────────────────────────────
log_step(2, "연속형 변수 분위별 Top3 Lift 분석")

NUMERIC_VARS = [
    "pthrRatg", "fe_horse_cum_win_rate", "fe_horse_cum_avg_rk",
    "fe_horse_race_count", "fe_horse_weight", "fe_weight_diff",
    "fe_ratg_per_weight", "fe_jcky_cum_win_rate", "fe_jcky_cum_top3_rate",
    "hrmJckyAlw", "fe_trar_cum_win_rate", "pthrBurdWgt", "pthrGtno",
    "fe_race_dist", "fe_track_humidity",
]
NUMERIC_VARS = [v for v in NUMERIC_VARS if v in df_v.columns]

all_lift_rows = []

for var in NUMERIC_VARS:
    if not pd.api.types.is_numeric_dtype(df_v[var]):
        continue

    # 5분위 구간 생성
    # fe_horse_cum_avg_rk: 낮을수록 좋음 → 방향 명시
    q_col = f"__q_{var}"
    df_v[q_col] = make_quantile_col(df_v[var], q=5, prefix="Q")

    lift = compute_lift(df_v, q_col, BASELINE)
    lift["feature"] = var
    lift = lift.rename(columns={q_col: "group"})

    # 원래 분위 범위 부연설명
    if var == "fe_horse_cum_avg_rk":
        lift["direction_note"] = "낮을수록 좋음 (Q1=실력 최상위 아님 주의, Q5가 높은 순위 기록)"

    all_lift_rows.append(lift)
    plot_lift_bar(lift.rename(columns={"group": q_col}), var, BASELINE, OUT_FIGS)
    log(f"  [{var}] 최고 Lift: {lift['lift'].max():.3f} (n={lift['n_entries'].max()})")

    # 임시 컬럼 제거
    df_v.drop(columns=[q_col], inplace=True)


# ─────────────────────────────────────────────
# Step 3: 범주형 변수 분석
# ─────────────────────────────────────────────
log_step(3, "범주형 변수 Top3 Lift 분석")

CATEG_VARS = [
    "cndRaceClas", "cndBurdGb", "cndRatg",
    "rsutTrckStus", "rsutWetr",
]
CATEG_VARS = [v for v in CATEG_VARS if v in df_v.columns]

for var in CATEG_VARS:
    lift = compute_lift(df_v, var, BASELINE)
    lift = lift.rename(columns={var: "group"})
    lift["feature"] = var
    all_lift_rows.append(lift)
    plot_lift_bar(lift.rename(columns={"group": var}), var, BASELINE, OUT_FIGS)
    log(f"  [{var}] 최고 Lift: {lift['lift'].max():.3f}")


# ─────────────────────────────────────────────
# Step 4: 전체 Lift 통합 저장
# ─────────────────────────────────────────────
log_step(4, "전체 변수 Lift 통합 저장")

lift_all = pd.concat(all_lift_rows, ignore_index=True)

# 컬럼 정리
col_order = ["feature", "group", "n_entries", "n_top3", "top3_rate",
             "diff_from_base", "lift", "note"]
col_order = [c for c in col_order if c in lift_all.columns]
lift_all = lift_all[col_order + [c for c in lift_all.columns if c not in col_order]]

lift_all.to_csv(
    os.path.join(OUT_TABLES, "feature_bucket_top3_lift.csv"),
    index=False, encoding="utf-8-sig"
)
log(f"  feature_bucket_top3_lift.csv 저장 완료 ({len(lift_all)}행)")


# ─────────────────────────────────────────────
# Step 5: 거리별 세그먼트 신호 분석
# ─────────────────────────────────────────────
log_step(5, "거리별 세그먼트 변수 효과 분석")

KEY_VARS_FOR_SEGMENT = [
    ("pthrRatg",            "레이팅"),
    ("fe_jcky_cum_win_rate","기수승률"),
    ("fe_horse_cum_avg_rk", "말평균순위"),
]
KEY_VARS_FOR_SEGMENT = [(v, lbl) for v, lbl in KEY_VARS_FOR_SEGMENT if v in df_v.columns]

dist_rows = []
dist_baseline = df_v.groupby("fe_race_dist")["target_is_top3"].mean().reset_index()
dist_baseline.columns = ["fe_race_dist", "dist_baseline"]

for dist_val, dist_grp in df_v.groupby("fe_race_dist"):
    n_dist = len(dist_grp)
    dist_base = dist_grp["target_is_top3"].mean()
    row = {
        "fe_race_dist": dist_val,
        "n_entries": n_dist,
        "dist_baseline": round(dist_base, 4),
    }

    for var, lbl in KEY_VARS_FOR_SEGMENT:
        if var not in dist_grp.columns:
            continue
        # 상위 33% (Q5 기준)
        thresh = dist_grp[var].quantile(0.67)
        if var == "fe_horse_cum_avg_rk":
            # 낮을수록 좋음 → 하위 33%가 상위권
            top_grp = dist_grp[dist_grp[var] <= dist_grp[var].quantile(0.33)]
        else:
            top_grp = dist_grp[dist_grp[var] >= thresh]

        if len(top_grp) < 30:
            row[f"{lbl}_top3_rate"] = None
            row[f"{lbl}_lift"] = None
        else:
            top3_rate = top_grp["target_is_top3"].mean()
            row[f"{lbl}_top3_rate"] = round(top3_rate, 4)
            row[f"{lbl}_lift"] = round(top3_rate / dist_base if dist_base > 0 else 0, 3)

    dist_rows.append(row)

dist_seg_df = pd.DataFrame(dist_rows).sort_values("fe_race_dist")
dist_seg_df.to_csv(
    os.path.join(OUT_TABLES, "distance_segment_signal.csv"),
    index=False, encoding="utf-8-sig"
)
log(f"  distance_segment_signal.csv 저장 완료 ({len(dist_seg_df)}개 거리)")


# ─────────────────────────────────────────────
# Step 6: 등급별 세그먼트 신호 분석
# ─────────────────────────────────────────────
log_step(6, "등급별 세그먼트 변수 효과 분석")

class_rows = []
for clas_val, clas_grp in df_v.groupby("cndRaceClas"):
    n_clas = len(clas_grp)
    clas_base = clas_grp["target_is_top3"].mean()
    row = {
        "cndRaceClas": clas_val,
        "n_entries": n_clas,
        "class_baseline": round(clas_base, 4),
    }

    for var, lbl in KEY_VARS_FOR_SEGMENT:
        if var not in clas_grp.columns:
            continue
        thresh = clas_grp[var].quantile(0.67)
        if var == "fe_horse_cum_avg_rk":
            top_grp = clas_grp[clas_grp[var] <= clas_grp[var].quantile(0.33)]
        else:
            top_grp = clas_grp[clas_grp[var] >= thresh]

        if len(top_grp) < 30:
            row[f"{lbl}_top3_rate"] = None
            row[f"{lbl}_lift"] = None
        else:
            top3_rate = top_grp["target_is_top3"].mean()
            row[f"{lbl}_top3_rate"] = round(top3_rate, 4)
            row[f"{lbl}_lift"] = round(top3_rate / clas_base if clas_base > 0 else 0, 3)

    class_rows.append(row)

class_seg_df = pd.DataFrame(class_rows).sort_values("class_baseline", ascending=False)
class_seg_df.to_csv(
    os.path.join(OUT_TABLES, "class_segment_signal.csv"),
    index=False, encoding="utf-8-sig"
)
log(f"  class_segment_signal.csv 저장 완료 ({len(class_seg_df)}개 등급)")


# ─────────────────────────────────────────────
# Step 7: 거리별 Lift 시각화 (요약)
# ─────────────────────────────────────────────
log_step(7, "거리별/등급별 요약 시각화")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("거리별 / 등급별 기준선 및 상위 변수 Lift", fontsize=13, fontweight="bold")

# 거리별
dist_plot = dist_seg_df.dropna(subset=["레이팅_lift"])
axes[0].bar(dist_plot["fe_race_dist"].astype(str), dist_plot["레이팅_lift"],
            color="#5B8DB8", label="레이팅 Lift")
axes[0].axhline(1.0, color="red", linestyle="--", label="기준선(Lift=1.0)")
axes[0].set_title("거리별 레이팅 상위 33% Lift")
axes[0].set_xlabel("거리(m)")
axes[0].set_ylabel("Lift")
axes[0].legend()

# 등급별
class_plot = class_seg_df.dropna(subset=["레이팅_lift"]).head(10)
axes[1].barh(class_plot["cndRaceClas"].astype(str),
             class_plot["레이팅_lift"], color="#E07B54")
axes[1].axvline(1.0, color="red", linestyle="--", label="기준선(Lift=1.0)")
axes[1].set_title("등급별 레이팅 상위 33% Lift")
axes[1].set_xlabel("Lift")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_FIGS, "segment_lift_summary.png"), dpi=120, bbox_inches="tight")
plt.close()
log("  segment_lift_summary.png 저장 완료")

print("\n[완료] 02_predictive_eda.py 실행 완료")
print(f"  - feature_bucket_top3_lift.csv ({len(lift_all)}행, {lift_all['feature'].nunique()}개 변수)")
print(f"  - distance_segment_signal.csv")
print(f"  - class_segment_signal.csv")
print(f"  - figures: {len([f for f in os.listdir(OUT_FIGS) if 'lift' in f.lower()])}개 저장")
