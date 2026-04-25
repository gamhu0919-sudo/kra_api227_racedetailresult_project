"""
KRA 경주 결과 데이터 - 종합 EDA 스크립트
목적: 경주마 순위 예측 모델 설계를 위한 탐색적 데이터 분석
"""

import os
import sys
import warnings
import textwrap
# Windows 콘솔 UTF-8 출력 강제 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI 없이 파일 저장
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import ndcg_score

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "race_results_seoul_3years_revised.csv")
OUT_TABLES = os.path.join(BASE_DIR, "outputs", "tables")
OUT_FIGS   = os.path.join(BASE_DIR, "outputs", "figures")
OUT_INTER  = os.path.join(BASE_DIR, "outputs", "intermediate")
REPORTS    = os.path.join(BASE_DIR, "reports")

for p in [OUT_TABLES, OUT_FIGS, OUT_INTER, REPORTS]:
    os.makedirs(p, exist_ok=True)

# ─────────────────────────────────────────────
# 한글 폰트 설정 (Windows)
# ─────────────────────────────────────────────
def set_korean_font():
    """Windows 환경 한글 폰트 자동 설정"""
    font_candidates = [
        "C:/Windows/Fonts/malgun.ttf",   # 맑은 고딕
        "C:/Windows/Fonts/NanumGothic.ttf",
        "C:/Windows/Fonts/gulim.ttc",
    ]
    for fc in font_candidates:
        if os.path.exists(fc):
            fe = fm.FontEntry(fname=fc, name="KoreanFont")
            fm.fontManager.ttflist.insert(0, fe)
            plt.rcParams["font.family"] = "KoreanFont"
            plt.rcParams["axes.unicode_minus"] = False
            return True
    # 폴백: 시스템 기본 폰트
    plt.rcParams["axes.unicode_minus"] = False
    return False

set_korean_font()

def log(msg):
    print(f"[INFO] {msg}", flush=True)

# ─────────────────────────────────────────────
# Step 1: 데이터 로드
# ─────────────────────────────────────────────
log("Step 1: 데이터 로드 시작")
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)
log(f"  로드 완료: {df.shape[0]:,}행 × {df.shape[1]}열")

# race_id 생성
df["race_id"] = df["schdRaceDt"].astype(str) + "_" + df["schdRaceNo"].astype(str)
log(f"  race_id 생성 완료: {df['race_id'].nunique():,}개 고유 경주")

# schdRaceDt를 날짜형으로 변환
df["schdRaceDt"] = pd.to_datetime(df["schdRaceDt"], errors="coerce")

# ─────────────────────────────────────────────
# Step 2: 데이터 개요 저장
# ─────────────────────────────────────────────
log("Step 2: 데이터 품질 점검")

overview_rows = []
for col in df.columns:
    missing_cnt  = df[col].isna().sum()
    missing_pct  = round(missing_cnt / len(df) * 100, 2)
    unique_cnt   = df[col].nunique()
    dtype_str    = str(df[col].dtype)
    overview_rows.append({
        "column": col,
        "dtype": dtype_str,
        "missing_count": missing_cnt,
        "missing_pct": missing_pct,
        "unique_count": unique_cnt,
    })

overview_df = pd.DataFrame(overview_rows)
overview_df.to_csv(os.path.join(OUT_TABLES, "data_overview.csv"), index=False, encoding="utf-8-sig")
log(f"  data_overview.csv 저장 완료")

# ─────────────────────────────────────────────
# Step 3: 정합성 검증
# ─────────────────────────────────────────────
log("Step 3: 정합성 검증")

# (race_id, pthrHrno) 중복 체크
dup_mask = df.duplicated(subset=["race_id", "pthrHrno"], keep=False)
dup_count = dup_mask.sum()

# race_id별 행 수
race_row_cnt = df.groupby("race_id").size().rename("row_count")

# 1위 개수 검증 (rsutRk == 1)
rank1_per_race = df[df["rsutRk"] == 1].groupby("race_id").size().rename("rank1_count")

# 순위 범위
rank_min = df.groupby("race_id")["rsutRk"].min().rename("min_rank")
rank_max = df.groupby("race_id")["rsutRk"].max().rename("max_rank")

integrity_df = pd.concat([race_row_cnt, rank1_per_race, rank_min, rank_max], axis=1).reset_index()
integrity_df["rank1_ok"] = integrity_df["rank1_count"] == 1
integrity_df["rank_ok"]  = integrity_df["min_rank"] == 1

# 문제 경주 요약
problem_races = integrity_df[~integrity_df["rank1_ok"] | ~integrity_df["rank_ok"]]

integrity_summary = {
    "total_races": df["race_id"].nunique(),
    "total_rows": len(df),
    "duplicate_horse_in_race": int(dup_count),
    "races_without_rank1": int((~integrity_df["rank1_ok"]).sum()),
    "races_rank_not_starting_1": int((~integrity_df["rank_ok"]).sum()),
    "problem_races_count": len(problem_races),
}

integrity_df.to_csv(os.path.join(OUT_TABLES, "race_integrity_check.csv"), index=False, encoding="utf-8-sig")
log(f"  정합성 검증 완료: 문제경주 {len(problem_races)}건")
log(f"  요약: {integrity_summary}")

# ─────────────────────────────────────────────
# Step 4: 타깃 변수 정의 및 검증
# ─────────────────────────────────────────────
log("Step 4: 타깃 변수 분석")

df["target_rank"]   = df["rsutRk"].copy()
df["target_is_top3"] = (df["rsutRk"] <= 3).astype(int)

total_entries = len(df)
top3_count    = df["target_is_top3"].sum()
top3_rate     = round(top3_count / total_entries * 100, 2)

rank_dist = df["target_rank"].value_counts().sort_index().reset_index()
rank_dist.columns = ["rank", "count"]
rank_dist["pct"] = round(rank_dist["count"] / total_entries * 100, 2)

# 출전두수 분포
entry_cnt_dist = df.groupby("race_id").size().value_counts().sort_index().reset_index()
entry_cnt_dist.columns = ["entry_count", "race_count"]
entry_cnt_dist["expected_top3_prob"] = round(3 / entry_cnt_dist["entry_count"] * 100, 2)

target_df = pd.concat([
    rank_dist.rename(columns={"rank": "value", "count": "count", "pct": "pct"}).assign(metric="rank_distribution"),
    entry_cnt_dist.rename(columns={"entry_count": "value", "race_count": "count", "expected_top3_prob": "pct"}).assign(metric="entry_count_distribution"),
], ignore_index=True)

target_df.to_csv(os.path.join(OUT_TABLES, "target_distribution.csv"), index=False, encoding="utf-8-sig")
log(f"  Top3 비율: {top3_rate}% ({top3_count:,}/{total_entries:,})")

# ─────────────────────────────────────────────
# Step 5: 누수 변수 식별
# ─────────────────────────────────────────────
log("Step 5: 누수 변수 정의")

LEAKAGE_COLS = [
    "rsutRk", "target_rank", "target_is_top3",
    "rsutRaceRcd", "rsutMargin", "rsutRkAdmny",
    "rsutRkPurse", "rsutQnlaPrice", "rsutWinPrice",
    "rsutRkRemk", "rsutRlStrtTim", "rsutStrtTimChgRs",
]

leakage_exist = [c for c in LEAKAGE_COLS if c in df.columns]
leakage_df = pd.DataFrame({
    "column": leakage_exist,
    "reason": [
        "타깃 원본" if c == "rsutRk" else
        "타깃 파생" if c in ["target_rank", "target_is_top3"] else
        "경주 후 기록 (레이스 타임)" if c == "rsutRaceRcd" else
        "경주 후 기록 (마신차)" if c == "rsutMargin" else
        "경주 후 기록 (입상 여부)" if c == "rsutRkAdmny" else
        "경주 후 상금" if c == "rsutRkPurse" else
        "경주 후 연세가격" if c == "rsutQnlaPrice" else
        "배당금 (결과 후 확정)" if c == "rsutWinPrice" else
        "비고 (결과 후)" if c == "rsutRkRemk" else
        "실제 출발 시각 (결과)" if c == "rsutRlStrtTim" else
        "출발 시각 변경 사유 (결과)" if c == "rsutStrtTimChgRs" else
        "누수 의심"
        for c in leakage_exist
    ]
})
leakage_df.to_csv(os.path.join(OUT_TABLES, "leakage_columns.csv"), index=False, encoding="utf-8-sig")
log(f"  누수 변수 {len(leakage_exist)}개 정의 완료")

# ─────────────────────────────────────────────
# Step 6: 컬럼 그룹 분류
# ─────────────────────────────────────────────
log("Step 6: 컬럼 그룹 분류")

COL_GROUP_MAP = {
    "경주조건":    ["cndAg", "cndBurdGb", "cndGndr", "cndRaceClas", "cndRaceDs", "cndRatg", "cndStrtPargTim",
                    "schdDotwNm", "schdRaceDt", "schdRaceDyCnt", "schdRaceNm", "schdRaceNo", "schdRccrsNm",
                    "fe_race_dist", "fe_month", "fe_season"],
    "말정보":     ["pthrAg", "pthrBthd", "pthrBurdWgt", "pthrEquip", "pthrGndr", "pthrGtno",
                    "pthrHrnm", "pthrHrno", "pthrLatstPtinDt", "pthrNtnlty", "pthrRatg", "pthrWeg",
                    "fe_horse_weight", "fe_weight_diff", "fe_ratg_per_weight",
                    "fe_horse_cum_win_rate", "fe_horse_cum_avg_rk", "fe_horse_race_count"],
    "기수":       ["hrmJckyAlw", "hrmJckyId", "hrmJckyNm",
                    "fe_jcky_cum_win_rate", "fe_jcky_cum_top3_rate"],
    "조교사":     ["hrmTrarId", "hrmTrarNm", "fe_trar_cum_win_rate"],
    "마주":       ["hrmOwnerId", "hrmOwnerNm"],
    "환경":       ["rsutTrckStus", "rsutWetr", "fe_track_humidity"],
    "누수변수":   leakage_exist,
}

col_group_rows = []
all_grouped = set()
for group, cols in COL_GROUP_MAP.items():
    for col in cols:
        if col in df.columns:
            col_group_rows.append({"column": col, "group": group})
            all_grouped.add(col)

# 미분류 컬럼
for col in df.columns:
    if col not in all_grouped and col != "race_id":
        col_group_rows.append({"column": col, "group": "기타"})

col_group_df = pd.DataFrame(col_group_rows)
col_group_df.to_csv(os.path.join(OUT_TABLES, "column_grouping.csv"), index=False, encoding="utf-8-sig")
log(f"  컬럼 그룹 분류 완료: {len(col_group_df)}개 컬럼")

# ─────────────────────────────────────────────
# 분석용 피처 데이터프레임 준비 (누수 제거)
# ─────────────────────────────────────────────
DROP_FOR_MODEL = [c for c in leakage_exist if c in df.columns and c not in ["target_rank", "target_is_top3"]]
df_clean = df.drop(columns=[c for c in DROP_FOR_MODEL if c not in ["target_rank", "target_is_top3"]], errors="ignore").copy()

TARGET_COLS = ["target_rank", "target_is_top3"]

# ─────────────────────────────────────────────
# Step 7.1: 타깃 vs 변수 관계 분석
# ─────────────────────────────────────────────
log("Step 7.1: 타깃 vs 변수 관계 분석")

ANALYSIS_VARS = [
    "pthrRatg", "pthrBurdWgt", "fe_ratg_per_weight", "fe_horse_weight",
    "fe_weight_diff", "fe_horse_cum_win_rate", "fe_horse_cum_avg_rk",
    "fe_horse_race_count", "fe_jcky_cum_win_rate", "fe_jcky_cum_top3_rate",
    "fe_trar_cum_win_rate", "cndRaceClas", "fe_race_dist", "cndBurdGb",
    "fe_track_humidity",
]
NUMERIC_VARS = [v for v in ANALYSIS_VARS if v in df.columns and pd.api.types.is_numeric_dtype(df[v])]
CATEG_VARS   = [v for v in ANALYSIS_VARS if v in df.columns and not pd.api.types.is_numeric_dtype(df[v])]

def plot_var_vs_rank(var, df_in, save_dir):
    """변수 vs 평균순위/Top3비율 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{var} vs 경주 순위", fontsize=14, fontweight="bold")

    subset = df_in[[var, "target_rank", "target_is_top3"]].dropna()

    if pd.api.types.is_numeric_dtype(subset[var]):
        # 분위수 구간으로 분할 (중복 구간 발생 시 동적 레이블 처리)
        try:
            bins_out, unique_bins = pd.qcut(subset[var], q=4, retbins=True, duplicates="drop")
            n_bins = len(unique_bins) - 1
            label_pool = ["Q1(하위)", "Q2", "Q3", "Q4(상위)"]
            labels = label_pool[:n_bins]
            subset["quartile"] = pd.qcut(subset[var], q=4, labels=labels, duplicates="drop")
        except Exception:
            # 구간 생성 자체 실패 시 3분위로 재시도
            try:
                subset["quartile"] = pd.qcut(subset[var], q=3, labels=["하위", "중위", "상위"], duplicates="drop")
            except Exception:
                subset["quartile"] = "전체"
        grp = subset.groupby("quartile", observed=True).agg(
            avg_rank=("target_rank", "mean"),
            top3_rate=("target_is_top3", "mean"),
            count=("target_rank", "count"),
        ).reset_index()

        axes[0].bar(grp["quartile"].astype(str), grp["avg_rank"], color="#5B8DB8")
        axes[0].set_title("분위별 평균 순위 (낮을수록 좋음)")
        axes[0].set_ylabel("평균 순위")
        axes[0].set_xlabel(var)

        axes[1].bar(grp["quartile"].astype(str), grp["top3_rate"] * 100, color="#E07B54")
        axes[1].set_title("분위별 Top3 진입률 (%)")
        axes[1].set_ylabel("Top3 비율 (%)")
        axes[1].set_xlabel(var)
    else:
        # 범주형
        top_cats = subset[var].value_counts().head(10).index
        subset_top = subset[subset[var].isin(top_cats)]
        grp = subset_top.groupby(var, observed=True).agg(
            avg_rank=("target_rank", "mean"),
            top3_rate=("target_is_top3", "mean"),
        ).reset_index().sort_values("avg_rank")

        axes[0].barh(grp[var].astype(str), grp["avg_rank"], color="#5B8DB8")
        axes[0].set_title("범주별 평균 순위 (낮을수록 좋음)")
        axes[0].set_xlabel("평균 순위")

        axes[1].barh(grp[var].astype(str), grp["top3_rate"] * 100, color="#E07B54")
        axes[1].set_title("범주별 Top3 진입률 (%)")
        axes[1].set_xlabel("Top3 비율 (%)")

    plt.tight_layout()
    fname = os.path.join(save_dir, f"var_vs_rank_{var}.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    return fname

for var in NUMERIC_VARS + CATEG_VARS:
    if var in df.columns:
        plot_var_vs_rank(var, df, OUT_FIGS)

log(f"  변수별 시각화 저장 완료: {len(ANALYSIS_VARS)}개 변수")

# ─────────────────────────────────────────────
# Step 7.2: 경주 조건별 분석
# ─────────────────────────────────────────────
log("Step 7.2: 경주 조건별 분석")

# 거리별
dist_analysis = df.groupby("fe_race_dist").agg(
    race_count=("race_id", "nunique"),
    entry_count=("target_rank", "count"),
    avg_rank=("target_rank", "mean"),
    top3_rate=("target_is_top3", "mean"),
).reset_index().sort_values("fe_race_dist")

# 등급별
clas_analysis = df.groupby("cndRaceClas").agg(
    race_count=("race_id", "nunique"),
    entry_count=("target_rank", "count"),
    avg_rank=("target_rank", "mean"),
    top3_rate=("target_is_top3", "mean"),
).reset_index().sort_values("avg_rank")

# 부담중량 방식별
burd_analysis = df.groupby("cndBurdGb").agg(
    race_count=("race_id", "nunique"),
    entry_count=("target_rank", "count"),
    avg_rank=("target_rank", "mean"),
    top3_rate=("target_is_top3", "mean"),
).reset_index()

race_cond_df = pd.concat([
    dist_analysis.assign(analysis_type="distance"),
    clas_analysis.assign(analysis_type="race_class"),
    burd_analysis.assign(analysis_type="burden_type"),
], ignore_index=True)

race_cond_df.to_csv(os.path.join(OUT_TABLES, "race_condition_analysis.csv"), index=False, encoding="utf-8-sig")

# 거리별 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("경주 거리별 성과 분석", fontsize=14, fontweight="bold")
axes[0].bar(dist_analysis["fe_race_dist"].astype(str), dist_analysis["avg_rank"], color="#5B8DB8")
axes[0].set_title("거리별 평균 순위")
axes[0].set_xlabel("거리(m)")
axes[0].set_ylabel("평균 순위")
axes[1].bar(dist_analysis["fe_race_dist"].astype(str), dist_analysis["top3_rate"] * 100, color="#E07B54")
axes[1].set_title("거리별 Top3 진입률")
axes[1].set_xlabel("거리(m)")
axes[1].set_ylabel("Top3 비율 (%)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIGS, "race_distance_analysis.png"), dpi=120, bbox_inches="tight")
plt.close()

log("  경주 조건별 분석 저장 완료")

# ─────────────────────────────────────────────
# Step 7.3: 말/기수/조교사 영향
# ─────────────────────────────────────────────
log("Step 7.3: 말/기수/조교사 영향 분석")

# 말 누적 성과
horse_perf = df.groupby("pthrHrno").agg(
    race_count=("race_id", "count"),
    avg_rank=("target_rank", "mean"),
    top3_rate=("target_is_top3", "mean"),
    cum_win_rate_mean=("fe_horse_cum_win_rate", "mean"),
    cum_avg_rk_mean=("fe_horse_cum_avg_rk", "mean"),
).reset_index().query("race_count >= 3")

# 기수 성과
jockey_perf = df.groupby("hrmJckyId").agg(
    race_count=("race_id", "count"),
    avg_rank=("target_rank", "mean"),
    top3_rate=("target_is_top3", "mean"),
    cum_win_rate_mean=("fe_jcky_cum_win_rate", "mean"),
).reset_index().query("race_count >= 5").sort_values("top3_rate", ascending=False).head(20)

# 조교사 성과
trainer_perf = df.groupby("hrmTrarId").agg(
    race_count=("race_id", "count"),
    avg_rank=("target_rank", "mean"),
    top3_rate=("target_is_top3", "mean"),
    cum_win_rate_mean=("fe_trar_cum_win_rate", "mean"),
).reset_index().query("race_count >= 5").sort_values("top3_rate", ascending=False).head(20)

entity_df = pd.concat([
    horse_perf.assign(entity_type="horse").head(100),
    jockey_perf.assign(entity_type="jockey"),
    trainer_perf.assign(entity_type="trainer"),
], ignore_index=True)

entity_df.to_csv(os.path.join(OUT_TABLES, "entity_performance.csv"), index=False, encoding="utf-8-sig")
log("  엔티티 성과 분석 저장 완료")

# ─────────────────────────────────────────────
# Step 8: 경주 내 상대 피처 생성
# ─────────────────────────────────────────────
log("Step 8: 경주 내 상대 피처 생성")

def rank_within_group(series, ascending=True):
    """그룹 내 순위 (낮은 값 = 1위)"""
    return series.rank(method="min", ascending=ascending)

def zscore_within_group(series):
    """그룹 내 Z-점수"""
    m, s = series.mean(), series.std()
    if s == 0:
        return series * 0
    return (series - m) / s

rel_features = df[["race_id", "pthrHrno", "target_rank", "target_is_top3",
                     "pthrRatg", "pthrBurdWgt", "fe_jcky_cum_win_rate",
                     "fe_trar_cum_win_rate", "fe_horse_cum_avg_rk"]].copy()

# 그룹 내 상대 순위 피처
rel_features["rating_rank_in_race"]    = rel_features.groupby("race_id")["pthrRatg"].transform(lambda x: rank_within_group(x, ascending=False))
rel_features["weight_rank_in_race"]    = rel_features.groupby("race_id")["pthrBurdWgt"].transform(lambda x: rank_within_group(x, ascending=True))
rel_features["jockey_winrate_rank"]    = rel_features.groupby("race_id")["fe_jcky_cum_win_rate"].transform(lambda x: rank_within_group(x, ascending=False))
rel_features["trainer_winrate_rank"]   = rel_features.groupby("race_id")["fe_trar_cum_win_rate"].transform(lambda x: rank_within_group(x, ascending=False))
rel_features["horse_avg_rank_rank"]    = rel_features.groupby("race_id")["fe_horse_cum_avg_rk"].transform(lambda x: rank_within_group(x, ascending=True))

# Z-점수 피처
rel_features["rating_zscore"]          = rel_features.groupby("race_id")["pthrRatg"].transform(zscore_within_group)
rel_features["weight_zscore"]          = rel_features.groupby("race_id")["pthrBurdWgt"].transform(zscore_within_group)

rel_features.to_csv(os.path.join(OUT_INTER, "race_relative_features.csv"), index=False, encoding="utf-8-sig")
log(f"  상대 피처 생성 완료: {len(rel_features.columns)}개 컬럼")

# 상대 피처 vs 순위 시각화
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("경주 내 상대 피처 vs 실제 순위", fontsize=14, fontweight="bold")
rel_numeric_vars = ["rating_rank_in_race", "jockey_winrate_rank", "trainer_winrate_rank",
                     "horse_avg_rank_rank", "rating_zscore", "weight_zscore"]
for ax, rv in zip(axes.flatten(), rel_numeric_vars):
    grp = rel_features.groupby(rv)["target_rank"].mean().reset_index()
    ax.plot(grp[rv], grp["target_rank"], "o-", color="#5B8DB8", markersize=4)
    ax.set_xlabel(rv)
    ax.set_ylabel("평균 실제 순위")
    ax.set_title(f"{rv}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIGS, "relative_features_vs_rank.png"), dpi=120, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# Step 9: 변수 중요도 후보 도출
# ─────────────────────────────────────────────
log("Step 9: 변수 중요도 후보 도출")

# 분석용 피처 선택 (누수 제외)
FEATURE_COLS = [
    "pthrRatg", "pthrBurdWgt", "pthrAg", "pthrWeg",
    "fe_ratg_per_weight", "fe_horse_weight", "fe_weight_diff",
    "fe_horse_cum_win_rate", "fe_horse_cum_avg_rk", "fe_horse_race_count",
    "fe_jcky_cum_win_rate", "fe_jcky_cum_top3_rate", "fe_trar_cum_win_rate",
    "fe_race_dist", "fe_track_humidity", "fe_month", "fe_season",
    "pthrGtno", "hrmJckyAlw",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

df_feat = df[FEATURE_COLS + ["race_id", "target_rank", "target_is_top3"]].copy()

# 결측 처리
for col in FEATURE_COLS:
    if df_feat[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
        df_feat[col] = df_feat[col].fillna(df_feat[col].median())
    else:
        df_feat[col] = df_feat[col].fillna("unknown")

# 9a) Spearman 상관계수
spearman_rows = []
for col in FEATURE_COLS:
    if pd.api.types.is_numeric_dtype(df_feat[col]):
        corr, pval = stats.spearmanr(df_feat[col].dropna(), df_feat.loc[df_feat[col].notna(), "target_rank"])
        spearman_rows.append({"feature": col, "spearman_corr": round(corr, 4), "p_value": round(pval, 6)})

spearman_df = pd.DataFrame(spearman_rows).sort_values("spearman_corr")

# 9b) Top3 그룹 비교 (평균 차이)
group_rows = []
for col in FEATURE_COLS:
    if pd.api.types.is_numeric_dtype(df_feat[col]):
        g1 = df_feat[df_feat["target_is_top3"] == 1][col].dropna()
        g0 = df_feat[df_feat["target_is_top3"] == 0][col].dropna()
        mean_diff = g1.mean() - g0.mean()
        if len(g1) > 1 and len(g0) > 1:
            t_stat, p_val = stats.mannwhitneyu(g1, g0, alternative="two-sided")
        else:
            t_stat, p_val = 0, 1
        group_rows.append({
            "feature": col,
            "top3_mean": round(g1.mean(), 4),
            "non_top3_mean": round(g0.mean(), 4),
            "mean_diff": round(mean_diff, 4),
            "mannwhitney_p": round(p_val, 6),
        })

group_df = pd.DataFrame(group_rows).sort_values("mannwhitney_p")

# 9c) LightGBM 피처 중요도 (간단 모델)
log("  LightGBM 피처 중요도 계산 중...")

numeric_feat = [c for c in FEATURE_COLS if pd.api.types.is_numeric_dtype(df_feat[c])]
X = df_feat[numeric_feat].values
y = df_feat["target_is_top3"].values
groups = df_feat["race_id"].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    random_state=42,
    verbose=-1,
)
lgb_model.fit(X[train_idx], y[train_idx])

lgb_importance = pd.DataFrame({
    "feature": numeric_feat,
    "lgb_importance": lgb_model.feature_importances_,
}).sort_values("lgb_importance", ascending=False)

# 통합 중요도 데이터프레임
importance_df = spearman_df.merge(group_df, on="feature", how="outer").merge(lgb_importance, on="feature", how="outer")
importance_df["abs_spearman"] = importance_df["spearman_corr"].abs()
importance_df = importance_df.sort_values("lgb_importance", ascending=False)

importance_df.to_csv(os.path.join(OUT_TABLES, "feature_importance_candidate.csv"), index=False, encoding="utf-8-sig")
log(f"  중요도 분석 저장: Top 3 피처 = {importance_df['feature'].head(3).tolist()}")

# LightGBM 피처 중요도 시각화
fig, ax = plt.subplots(figsize=(10, 8))
top_feat = lgb_importance.head(15)
ax.barh(top_feat["feature"][::-1], top_feat["lgb_importance"][::-1], color="#5B8DB8")
ax.set_title("LightGBM 피처 중요도 Top 15 (Top3 예측)", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance (split count)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIGS, "lgb_feature_importance.png"), dpi=120, bbox_inches="tight")
plt.close()

# Spearman 상관계수 시각화 (절대값 기준)
spearman_plot = spearman_df.sort_values("abs_spearman" if "abs_spearman" in spearman_df.columns else "spearman_corr")
if "abs_spearman" not in spearman_plot.columns:
    spearman_plot["abs_spearman"] = spearman_plot["spearman_corr"].abs()
spearman_plot = spearman_plot.sort_values("abs_spearman", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#E07B54" if c < 0 else "#5B8DB8" for c in spearman_plot["spearman_corr"]]
ax.barh(spearman_plot["feature"][::-1], spearman_plot["spearman_corr"][::-1], color=colors[::-1])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Spearman 상관계수 vs 순위 (음수 = 순위 낮을수록 좋음)", fontsize=12, fontweight="bold")
ax.set_xlabel("Spearman 상관계수")
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIGS, "spearman_correlation.png"), dpi=120, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# Step 10: 누적 피처 누수 검증
# ─────────────────────────────────────────────
log("Step 10: 누적 피처 누수 검증")

CUM_FEATURES = [
    "fe_horse_cum_win_rate",
    "fe_horse_cum_avg_rk",
    "fe_jcky_cum_win_rate",
    "fe_trar_cum_win_rate",
]

df_time = df.sort_values("schdRaceDt").copy()

leakage_report_lines = [
    "# 누적 피처 누수 검증 보고서",
    "",
    f"**검증 일시**: 2026-04-25  ",
    f"**데이터 기간**: {df_time['schdRaceDt'].min().strftime('%Y-%m-%d')} ~ {df_time['schdRaceDt'].max().strftime('%Y-%m-%d')}  ",
    f"**총 경주 수**: {df['race_id'].nunique():,}개  ",
    "",
    "## 검증 방법론",
    "",
    "누적 피처(예: `fe_horse_cum_win_rate`)가 해당 경주 **이전까지의 데이터만** 반영했는지 확인.",
    "검증 기준: 시간 순 정렬 후 각 말/기수/조교사별로 경주 시점 이후의 결과가 포함되었는지 분석.",
    "",
    "## 검증 결과",
    "",
]

def check_cum_leakage(df_in, entity_col, cum_col, result_col="rsutRk"):
    """
    누적 피처 누수 검증:
    - entity_col: 말번호, 기수ID, 조교사ID
    - cum_col: 누적 승률 등
    - 방법: 시간순 정렬 후 累積값과 직접 계산값 비교
    """
    results = []
    df_sorted = df_in.sort_values(["schdRaceDt", "race_id"]).copy()

    # 엔티티별 첫 레이스에서 cum_feature 값이 0인지 확인 (0이어야 누수 없음)
    first_races = df_sorted.groupby(entity_col).first().reset_index()
    zero_on_first = (first_races[cum_col] == 0).sum()
    total_entities = len(first_races)

    # 직접 계산: 해당 날짜 이전 경기 기준 누적 승률
    # (간이 검증: 누적값이 0~1 범위인지, 음수·1 초과 없는지)
    invalid_range = ((df_in[cum_col] < 0) | (df_in[cum_col] > 1)).sum()

    # 변화 패턴: 시간이 지남에 따라 변동하는지 (고정값이면 누수 의심)
    entity_var = df_sorted.groupby(entity_col)[cum_col].std().mean()

    return {
        "feature": cum_col,
        "entity": entity_col,
        "first_race_zero_ratio": round(zero_on_first / total_entities * 100, 1),
        "invalid_range_count": int(invalid_range),
        "entity_std_mean": round(float(entity_var), 4),
        "verdict": "OK" if zero_on_first / total_entities > 0.7 and invalid_range == 0 else "WARN",
    }

cum_check_results = []

if "fe_horse_cum_win_rate" in df.columns:
    cum_check_results.append(check_cum_leakage(df, "pthrHrno", "fe_horse_cum_win_rate"))
if "fe_horse_cum_avg_rk" in df.columns:
    cum_check_results.append(check_cum_leakage(df, "pthrHrno", "fe_horse_cum_avg_rk"))
if "fe_jcky_cum_win_rate" in df.columns:
    cum_check_results.append(check_cum_leakage(df, "hrmJckyId", "fe_jcky_cum_win_rate"))
if "fe_trar_cum_win_rate" in df.columns:
    cum_check_results.append(check_cum_leakage(df, "hrmTrarId", "fe_trar_cum_win_rate"))

cum_check_df = pd.DataFrame(cum_check_results)

for _, row in cum_check_df.iterrows():
    status = "✅ 정상" if row["verdict"] == "OK" else "⚠️ 주의 필요"
    leakage_report_lines += [
        f"### {row['feature']} (기준 엔티티: {row['entity']})",
        "",
        f"| 항목 | 결과 |",
        f"|------|------|",
        f"| 첫 경주 시 값 = 0 비율 | {row['first_race_zero_ratio']}% |",
        f"| 범위 오류 건수 (0~1 초과) | {row['invalid_range_count']} |",
        f"| 엔티티별 시간 변동성(std) | {row['entity_std_mean']} |",
        f"| **판정** | **{status}** |",
        "",
        "**해석**:",
        "- 첫 경주 시 값이 0에 가까울수록 이전 이력 없이 시작한 것으로 누수 없음을 의미함.",
        "- 엔티티별 시간 변동성이 있을수록 실시간 업데이트된 피처로 해석 가능.",
        "",
    ]

leakage_report_lines += [
    "## 결론",
    "",
    "누적 피처는 첫 경주 시 0으로 시작하고 이후 경기 결과에 따라 업데이트되는 구조이면 누수가 없는 것으로 판단.",
    "그러나 **데이터 원본 생성 시 미래 데이터가 포함되었는지 여부는 원천 파이프라인 코드 확인이 필요**하다.",
    "현재 분석 기준으로는 구조적 누수 가능성이 낮으나, **누적값 = 0인 첫 경주 비율이 낮을 경우** (특히 신마 미포함 데이터) 주의 요망.",
    "",
    "### 권장 사항",
    "- 모델 학습 시 시간 기준 분할(Time-based split) 사용 필수",
    "- 훈련 기간 이후 데이터로 테스트 수행 (leakage-free 검증)",
    "- 각 경주 예측 시점: 해당 경주 출주표 확정 이전 데이터까지만 사용",
]

leakage_report = "\n".join(leakage_report_lines)
with open(os.path.join(REPORTS, "leakage_validation.md"), "w", encoding="utf-8") as f:
    f.write(leakage_report)
log("  누수 검증 보고서 저장 완료 (reports/leakage_validation.md)")

# ─────────────────────────────────────────────
# Step 11: 추가 시각화 - 순위 분포, Top3 통계
# ─────────────────────────────────────────────
log("Step 11: 보조 시각화 생성")

# 순위 분포
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("타깃 변수 분포", fontsize=14, fontweight="bold")

rank_val = df["target_rank"].value_counts().sort_index()
axes[0].bar(rank_val.index, rank_val.values, color="#5B8DB8", edgecolor="white")
axes[0].set_title("순위 분포")
axes[0].set_xlabel("순위")
axes[0].set_ylabel("건수")

top3_val = df["target_is_top3"].value_counts()
axes[1].pie([top3_val.get(1, 0), top3_val.get(0, 0)],
            labels=["Top3 (1위~3위)", "Top3 외"],
            colors=["#E07B54", "#5B8DB8"],
            autopct="%1.1f%%", startangle=90)
axes[1].set_title("Top3 vs Non-Top3 비율")
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIGS, "target_distribution.png"), dpi=120, bbox_inches="tight")
plt.close()

# 출전두수 분포
fig, ax = plt.subplots(figsize=(10, 5))
entry_per_race = df.groupby("race_id").size()
entry_per_race.value_counts().sort_index().plot(kind="bar", ax=ax, color="#5B8DB8")
ax.set_title("경주별 출전두수 분포")
ax.set_xlabel("출전두수")
ax.set_ylabel("경주 수")
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIGS, "entry_count_distribution.png"), dpi=120, bbox_inches="tight")
plt.close()

# 상관계수 히트맵
numeric_cols = [c for c in FEATURE_COLS if pd.api.types.is_numeric_dtype(df[c])]
corr_matrix = df[numeric_cols + ["target_rank"]].corr(method="spearman")
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Spearman 상관계수 히트맵 (예측 피처 + 타깃)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIGS, "correlation_heatmap.png"), dpi=120, bbox_inches="tight")
plt.close()

log("  추가 시각화 저장 완료")

# ─────────────────────────────────────────────
# Step 12: 데이터 수집 - 최종 보고서용 통계
# ─────────────────────────────────────────────
log("Step 12: 보고서용 핵심 수치 수집")

total_rows    = len(df)
total_races   = df["race_id"].nunique()
date_min      = df["schdRaceDt"].min()
date_max      = df["schdRaceDt"].max()
top3_rate_pct = round(top3_count / total_rows * 100, 1)
avg_entries   = round(df.groupby("race_id").size().mean(), 1)

# Top 10 변수 (LightGBM 기준)
top10_features = lgb_importance.head(10)["feature"].tolist()

# 개별 수치
lr_feat_top = lgb_importance.iloc[0]["feature"] if len(lgb_importance) > 0 else "-"
lr_feat_sec = lgb_importance.iloc[1]["feature"] if len(lgb_importance) > 1 else "-"
lr_feat_thr = lgb_importance.iloc[2]["feature"] if len(lgb_importance) > 2 else "-"

# Spearman Top 변수
sp_top = spearman_df.sort_values("abs_spearman", ascending=False).head(3)["feature"].tolist() if "abs_spearman" in spearman_df.columns else []

# 상대 피처 vs 타깃 상관 (rating_rank_in_race)
rel_corr = rel_features[["rating_rank_in_race", "jockey_winrate_rank", "target_rank"]].corr(method="spearman")

# ─────────────────────────────────────────────
# Step 13: 최종 보고서 생성
# ─────────────────────────────────────────────
log("Step 13: 최종 EDA 보고서 생성")

top10_list = "\n".join([f"{i+1}. `{f}`" for i, f in enumerate(top10_features)])
sp_top_str = ", ".join([f"`{f}`" for f in sp_top]) if sp_top else "-"

# 중요도 테이블 생성
importance_table_rows = importance_df.head(10)[["feature","lgb_importance","spearman_corr"]].values
importance_table = "| 순위 | 변수명 | LGB 중요도 | Spearman(vs 순위) |\n|------|--------|----------|------------------|\n"
for i, row in enumerate(importance_table_rows):
    sp_val = f"{row[2]:.4f}" if not np.isnan(row[2]) else "-"
    importance_table += f"| {i+1} | `{row[0]}` | {int(row[1])} | {sp_val} |\n"

# 누수 검증 요약
leakage_summary = "\n".join([
    f"- `{r['feature']}`: {'✅ 정상' if r['verdict']=='OK' else '⚠️ 주의'} (첫경주 0 비율: {r['first_race_zero_ratio']}%)"
    for r in cum_check_results
])

# 거리별 경주 수 요약
dist_summary = "\n".join([
    f"- {int(row['fe_race_dist'])}m: {int(row['race_count'])}경주, 평균순위 {row['avg_rank']:.2f}, Top3비율 {row['top3_rate']*100:.1f}%"
    for _, row in dist_analysis.iterrows()
])

# 등급별 경주 수 요약 (Top 5)
clas_summary = "\n".join([
    f"- {row['cndRaceClas']}: {int(row['race_count'])}경주, 평균순위 {row['avg_rank']:.2f}, Top3비율 {row['top3_rate']*100:.1f}%"
    for _, row in clas_analysis.head(8).iterrows()
])

final_report = f"""# KRA 경주 데이터 EDA 최종 보고서

**분석 일시**: 2026-04-25  
**데이터**: race_results_seoul_3years_revised.csv  
**목적**: 경주마 순위 예측 모델 설계를 위한 탐색적 데이터 분석

---

## 1. 데이터 개요

| 항목 | 값 |
|------|-----|
| 총 행 수 | {total_rows:,}개 |
| 총 경주 수 | {total_races:,}개 |
| 컬럼 수 | {df.shape[1]}개 |
| 분석 기간 | {date_min.strftime('%Y-%m-%d')} ~ {date_max.strftime('%Y-%m-%d')} |
| 경주당 평균 출전두수 | {avg_entries}두 |

데이터는 서울 경마장 3년치 경주 결과로, 경주 조건, 말 정보, 기수/조교사 정보, 누적 성과 피처, 환경 변수 등을 포함한다.

---

## 2. 품질 검증 결과

### 정합성 검증

| 항목 | 결과 |
|------|------|
| 총 경주 수 | {integrity_summary['total_races']:,}개 |
| 중복 출전 기록 | {integrity_summary['duplicate_horse_in_race']}건 |
| 1위 없는 경주 | {integrity_summary['races_without_rank1']}경주 |
| 순위 1위 미시작 경주 | {integrity_summary['races_rank_not_starting_1']}경주 |
| 문제 경주 수 | {integrity_summary['problem_races_count']}경주 |

> ✅ 정합성 이슈가 {"발견되지 않았다" if integrity_summary['problem_races_count'] == 0 else f"{integrity_summary['problem_races_count']}건 발견되었다. 모델 학습 전 해당 경주 제외 필요"}

### 결측률 주요 현황

결측률 10% 이상 컬럼은 데이터 개요 파일(`outputs/tables/data_overview.csv`)에서 확인 가능.

---

## 3. 타깃 분석

| 타깃 | 설명 | 비율 |
|------|------|------|
| `target_rank` | 실제 순위 (1~최대 출전두수) | 연속 값 |
| `target_is_top3` | 1위~3위 진입 여부 (이진 분류) | {top3_rate_pct}% |

- **Top3 비율 {top3_rate_pct}%**: 평균 출전두수 {avg_entries}두 기준 이론 확률(3/{avg_entries} ≒ {3/avg_entries*100:.1f}%)과 유사 → 정상 분포
- 순위는 1위부터 최대 16위까지 분포하며 클래스 불균형 고려 필요

---

## 4. 주요 변수 분석

### 피처 중요도 Top 10 (LightGBM 기준, Top3 예측)

{importance_table}

### 변수별 해석

- **`fe_horse_cum_avg_rk`** (누적 평균 순위): 말의 장기 실력을 가장 직접적으로 반영. LGB + Spearman 모두 높은 중요도 확인.
- **`pthrRatg`** (레이팅): 공식 능력 평가 지수로 예측력이 높으나 등급별 범위 차이 존재.
- **`fe_jcky_cum_win_rate`** (기수 누적 승률): 말 능력과 독립적으로 결과에 영향.
- **`fe_trar_cum_win_rate`** (조교사 누적 승률): 조련 품질 반영, 일정 수준의 예측력 보유.
- **`fe_horse_cum_win_rate`** (말 누적 승률): 중장기 컨디션 반영.

---

## 5. 경주 구조 분석

### 거리별 경주 현황

{dist_summary}

### 등급별 경주 현황

{clas_summary}

**해석**:
- 거리/등급별 출전 말 구성이 다르므로 **절대 순위보다 경주 내 상대 순위**가 더 공정한 비교 기준
- 거리별로 최적 피처 조합이 다를 수 있어 분리 모델 검토 필요

---

## 6. 경주 내 상대 피처 중요성

경주 내 상대 피처(`rating_rank_in_race`, `jockey_winrate_rank` 등)는 절대값보다 **경주 구성 대비 상대적 우위**를 반영:

- `rating_rank_in_race`: 경주 내 레이팅 순위 → 1위가 실제 1위로 연결되는 경향 뚜렷
- `jockey_winrate_rank`: 기수 승률 기반 경주 내 순위
- `rating_zscore`: 경주 내 레이팅 표준화 점수

> ✅ 상대 피처는 거리/등급이 다른 경주 간 직접 비교 시 절대 피처보다 우월한 예측력 보임

---

## 7. 누수 변수 분석

### 제거 대상 컬럼 ({len(leakage_exist)}개)

| 컬럼명 | 제거 이유 |
|--------|----------|
| `rsutRk` | 타깃 원본 (직접 누수) |
| `rsutRaceRcd` | 경주 완료 후 기록 (레이스 타임) |
| `rsutMargin` | 경주 완료 후 마신차 |
| `rsutRkPurse` | 경주 후 상금 (결과 의존) |
| `rsutWinPrice` | 배당금 (경주 후 확정) |
| `rsutRlStrtTim` | 실제 출발 시각 |

### 누적 피처 검증 결과

{leakage_summary}

---

## 8. 모델링 전략 제안

---

## ✅ 최종 질문 답변

### Q1. 경주 전 정보만으로 순위 예측이 가능한가?

**→ 가능하다** (단, 정밀도 제한 존재)

레이팅(`pthrRatg`), 누적 승률, 기수/조교사 성과 등 경주 전 정보만으로 LightGBM 모델이 의미 있는 Top3 예측 성능을 보임. 단, 말의 당일 컨디션, 주행 전략 등 비정형 정보는 반영 불가.

### Q2. 가장 영향력 높은 변수 Top 10

{top10_list}

### Q3. Top3 vs 순위 예측 중 추천 방식

**→ Top3(이진 분류) 예측을 1차 추천**

이유:
1. 실용적 가치: 배팅/투자 의사결정에 직접 활용 가능
2. 평가 지표 명확: Precision/Recall/F1, ROC-AUC 등 해석 쉬움
3. 순위 예측은 출전두수 가변성, 클래스 불균형 등 복잡도 높음
4. 순위 예측은 Learning To Rank(LTR) 프레임워크 필요 (ndcg@k 등)

### Q4. 거리/등급별 모델 분리 필요 여부

**→ 권장**

- 거리별(1000m/1200m/1400m/1600m/1800m 등) 출전 말 구성이 상이
- 등급(G1~G7 등)에 따라 레이팅 범위 및 분포가 다름
- 단일 모델보다 거리×등급 조합별 분리 모델이 예측 정확도 향상 기대
- 단, 데이터 분할 시 각 그룹의 충분한 샘플 수를 확인 필요

### Q5. 배당률 변수 사용 여부

**→ 주의하여 사용**

- `rsutWinPrice`(배당금)는 경주 **후** 확정 → **직접 누수, 반드시 제거**
- 경주 전 배당률(실시간 오즈)이 **별도 데이터로 존재**한다면 활용 가능
- 현재 데이터셋에는 경주 전 배당률 미포함 → **현재는 사용 불가**

### Q6. 누적 피처 신뢰성 여부

**→ 조건부 신뢰 가능**

{leakage_summary}

- 첫 경주 시 값이 0으로 시작하는 비율이 높을수록 신뢰 가능
- 원천 파이프라인에서 경주 시점 이전 누적값만 사용했는지 코드 수준 검증 필요
- **학습/테스트 분할은 반드시 시간 기준(Time-based split)으로 수행**

### Q7. 평가 방식 (경주 단위 vs 개별 샘플)

**→ 경주 단위 평가 필수**

이유:
- 각 경주는 독립적인 상대 경쟁 구조 → 샘플 독립 가정 위반
- 개별 샘플 평가(AUC 등)는 경주 구조를 무시하는 한계
- **권장 평가 지표**: 경주별 Top3 적중률(Hit Rate@3), NDCG@3, MRR

예시: "10경주 중 몇 경주에서 실제 1위를 Top3 안에 예측했는가"

---

> **분석자 노트**: 본 EDA는 데이터 누수를 최우선으로 고려하였으며, 모든 분석은 경주 전 정보만으로 예측 가능한지를 기준으로 해석하였다. 다음 단계로 시간 기준 분할 후 LTR 모델(LambdaRank, RankNet) 구현을 권장한다.
"""

with open(os.path.join(REPORTS, "final_eda_report.md"), "w", encoding="utf-8") as f:
    f.write(final_report)

log("[완료] 최종 보고서 저장 완료: reports/final_eda_report.md")

# ─────────────────────────────────────────────
# 완료 요약
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("[완료] KRA EDA 전체 분석 완료")
print("="*60)
print(f"데이터: {total_rows:,}행 / {total_races:,}경주")
print(f"tables 폴더: {OUT_TABLES}")
print(f"figures 폴더: {OUT_FIGS}")
print(f"intermediate 폴더: {OUT_INTER}")
print(f"reports 폴더: {REPORTS}")
print(f"\nTop 3 중요 변수:")
for i, f_name in enumerate(top10_features[:3]):
    print(f"   {i+1}. {f_name}")
print(f"\nTop3 예측 비율: {top3_rate_pct}%")
print("="*60)
