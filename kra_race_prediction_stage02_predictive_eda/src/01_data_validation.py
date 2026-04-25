"""
01_data_validation.py
─────────────────────
데이터 로드 및 기본 품질 검증
- 기본 요약 통계
- 결측률 분석
- 컬럼 타입 분류
- race_id 생성 및 경주 단위 정합성 검증
- 타깃 기준선 계산 (산술 기준값과 비교)

산출물:
  outputs/tables/data_basic_summary.csv
  outputs/tables/missing_value_summary.csv
  outputs/tables/column_type_summary.csv
  outputs/tables/race_integrity_check.csv
  outputs/tables/problem_races.csv
  outputs/tables/target_baseline_check.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from config_loader import (
    RAW_CSV, SCHEMA_XL, OUT_TABLES, DATA_RAW,
    LEAKAGE_COLS, MIN_SAMPLE_N, log, log_step
)


# ─────────────────────────────────────────────
# Step 1: 데이터 로드
# ─────────────────────────────────────────────
log_step(1, "데이터 로드")

df = pd.read_csv(RAW_CSV, encoding="utf-8-sig", low_memory=False)
log(f"로드 완료: {df.shape[0]:,}행 × {df.shape[1]}열")

# 날짜 처리
df["schdRaceDt"] = pd.to_datetime(df["schdRaceDt"], errors="coerce")

# race_id 생성
df["race_id"] = df["schdRaceDt"].dt.strftime("%Y%m%d") + "_" + df["schdRaceNo"].astype(str).str.zfill(2)
log(f"race_id 생성: {df['race_id'].nunique():,}개 고유 경주")

# 타깃 생성
df["target_rank"]    = df["rsutRk"].copy()
df["target_is_top3"] = (df["rsutRk"] <= 3).astype(int)

# 원본 데이터 복사본 저장 (race_id, 타깃 포함)
df.to_csv(os.path.join(DATA_RAW, "race_data_with_ids.csv"), index=False, encoding="utf-8-sig")

# ─────────────────────────────────────────────
# Step 2: 기본 요약 통계
# ─────────────────────────────────────────────
log_step(2, "기본 요약 통계 생성")

n_rows      = len(df)
n_races     = df["race_id"].nunique()
n_cols      = df.shape[1]
date_min    = df["schdRaceDt"].min()
date_max    = df["schdRaceDt"].max()
avg_entries = round(df.groupby("race_id").size().mean(), 2)

basic_summary = pd.DataFrame([
    {"항목": "총 행 수",         "값": f"{n_rows:,}"},
    {"항목": "총 경주 수",       "값": f"{n_races:,}"},
    {"항목": "총 컬럼 수",       "값": f"{n_cols}"},
    {"항목": "분석 시작일",      "값": str(date_min.date())},
    {"항목": "분석 종료일",      "값": str(date_max.date())},
    {"항목": "분석 기간(일)",    "값": f"{(date_max - date_min).days}일"},
    {"항목": "경주당 평균 출전두수", "값": f"{avg_entries}두"},
    {"항목": "출전두수 최소",    "값": str(df.groupby("race_id").size().min())},
    {"항목": "출전두수 최대",    "값": str(df.groupby("race_id").size().max())},
])
basic_summary.to_csv(os.path.join(OUT_TABLES, "data_basic_summary.csv"), index=False, encoding="utf-8-sig")
log("  data_basic_summary.csv 저장 완료")

# ─────────────────────────────────────────────
# Step 3: 결측률 분석
# ─────────────────────────────────────────────
log_step(3, "결측률 분석")

missing_rows = []
for col in df.columns:
    n_miss = df[col].isna().sum()
    pct    = round(n_miss / n_rows * 100, 2)
    missing_rows.append({
        "column": col,
        "missing_count": n_miss,
        "missing_pct": pct,
        "status": "HIGH" if pct > 30 else ("MID" if pct > 10 else ("LOW" if pct > 0 else "NONE")),
    })

missing_df = pd.DataFrame(missing_rows).sort_values("missing_pct", ascending=False)
missing_df.to_csv(os.path.join(OUT_TABLES, "missing_value_summary.csv"), index=False, encoding="utf-8-sig")

high_missing = missing_df[missing_df["status"] == "HIGH"]["column"].tolist()
log(f"  결측률 30% 초과 컬럼: {high_missing}")
log(f"  missing_value_summary.csv 저장 완료")

# ─────────────────────────────────────────────
# Step 4: 컬럼 타입 분류
# ─────────────────────────────────────────────
log_step(4, "컬럼 타입 분류")

def classify_col(col, series):
    """컬럼을 역할별로 분류"""
    if col in LEAKAGE_COLS:
        return "누수_제거대상"
    if col in ["target_rank", "target_is_top3"]:
        return "타깃"
    if col in ["race_id", "schdRaceDt", "schdRaceNo", "pthrHrno", "pthrHrnm",
               "hrmJckyId", "hrmJckyNm", "hrmTrarId", "hrmTrarNm", "hrmOwnerId", "hrmOwnerNm"]:
        return "식별자"
    if col.startswith("fe_"):
        return "파생피처_경주전"
    if col.startswith("cnd"):
        return "경주조건"
    if col.startswith("pthr"):
        return "말정보"
    if col.startswith("hrm"):
        return "기수조교사"
    if col.startswith("schd"):
        return "경주일정"
    if col.startswith("rsut"):
        return "누수_경주후"
    return "기타"

type_rows = []
for col in df.columns:
    series = df[col]
    is_numeric = pd.api.types.is_numeric_dtype(series)
    n_unique   = series.nunique()
    role       = classify_col(col, series)
    type_rows.append({
        "column": col,
        "pandas_dtype": str(series.dtype),
        "is_numeric": is_numeric,
        "unique_count": n_unique,
        "role": role,
    })

type_df = pd.DataFrame(type_rows)
type_df.to_csv(os.path.join(OUT_TABLES, "column_type_summary.csv"), index=False, encoding="utf-8-sig")
log(f"  컬럼 타입 분류 완료: {type_df['role'].value_counts().to_dict()}")

# ─────────────────────────────────────────────
# Step 5: 경주 단위 정합성 검증
# ─────────────────────────────────────────────
log_step(5, "경주 단위 정합성 검증")

race_grp = df.groupby("race_id")

# 각 경주별 검증 지표 수집
integrity_rows = []
for race_id, grp in race_grp:
    n_entries  = len(grp)
    n_rank1    = (grp["rsutRk"] == 1).sum()
    n_top3     = (grp["rsutRk"] <= 3).sum()
    n_dup_horse= grp["pthrHrno"].duplicated().sum()
    rank_min   = grp["rsutRk"].min()
    rank_max   = grp["rsutRk"].max()
    
    # 실격/취소 코드: rsutRk >= 90은 결과 불명이므로 정상 범위에서 제외
    valid_ranks = grp[grp["rsutRk"] < 90]["rsutRk"]
    rank_out    = ((valid_ranks < 1) | (valid_ranks > n_entries)).sum()

    # 문제 여부 판정
    issues = []
    if n_rank1 != 1:
        issues.append(f"1위_수={n_rank1}")
    if n_dup_horse > 0:
        issues.append(f"말중복={n_dup_horse}")
    if rank_out > 0:
        issues.append(f"순위범위오류={rank_out}")
    if rank_min != 1:
        issues.append("순위1시작아님")

    integrity_rows.append({
        "race_id":    race_id,
        "n_entries":  n_entries,
        "n_rank1":    n_rank1,
        "n_top3":     n_top3,
        "n_dup_horse": n_dup_horse,
        "rank_min":   rank_min,
        "rank_max":   rank_max,
        "rank_out_of_range": rank_out,
        "issue_desc": "; ".join(issues) if issues else "",
        "is_valid":   len(issues) == 0,
    })

integrity_df = pd.DataFrame(integrity_rows)
integrity_df.to_csv(os.path.join(OUT_TABLES, "race_integrity_check.csv"), index=False, encoding="utf-8-sig")

# 문제 경주 별도 저장
problem_df = integrity_df[~integrity_df["is_valid"]].copy()
problem_df.to_csv(os.path.join(OUT_TABLES, "problem_races.csv"), index=False, encoding="utf-8-sig")

# is_valid_race 플래그를 원본 df에 합류
df = df.merge(integrity_df[["race_id", "is_valid"]].rename(columns={"is_valid": "is_valid_race"}),
              on="race_id", how="left")

n_valid   = integrity_df["is_valid"].sum()
n_invalid = (~integrity_df["is_valid"]).sum()
log(f"  정상 경주: {n_valid:,} / 문제 경주: {n_invalid} ({n_invalid/len(integrity_df)*100:.1f}%)")
log(f"  문제 유형: {problem_df['issue_desc'].value_counts().head().to_dict()}")

# ─────────────────────────────────────────────
# Step 6: 타깃 기준선 계산
# ─────────────────────────────────────────────
log_step(6, "타깃 기준선 계산 (산술 기준값 정의)")

# 유효 경주만 사용
df_valid = df[df["is_valid_race"]].copy()

total_top3_rate   = df_valid["target_is_top3"].mean()
avg_entries_valid = df_valid.groupby("race_id").size().mean()
arithmetic_base   = 3 / avg_entries_valid
diff_from_arith   = total_top3_rate - arithmetic_base

log(f"  전체 Top3 비율: {total_top3_rate:.4f} ({total_top3_rate*100:.2f}%)")
log(f"  평균 출전두수: {avg_entries_valid:.2f}두")
log(f"  산술 기준값(3/평균두수): {arithmetic_base:.4f} ({arithmetic_base*100:.2f}%)")
log(f"  실제-산술 차이: {diff_from_arith:.4f}p")

baseline_check = pd.DataFrame([
    {"항목": "유효 경주 총 출전 수",           "값": len(df_valid)},
    {"항목": "전체 Top3 수",                  "값": df_valid["target_is_top3"].sum()},
    {"항목": "전체 Top3 비율",                "값": f"{total_top3_rate*100:.4f}%"},
    {"항목": "평균 출전두수(유효 경주)",        "값": f"{avg_entries_valid:.4f}"},
    {"항목": "산술 기준값 (3 / 평균두수)",     "값": f"{arithmetic_base*100:.4f}%"},
    {"항목": "실제 Top3 - 산술 기준값 차이",   "값": f"{diff_from_arith*100:.4f}%p"},
    {"항목": "해석",
     "값": "전체 Top3 비율은 예측 인사이트가 아니라 타깃 생성이 정상인지 확인하는 기준선이다. "
           "실제 예측 가능성은 특정 변수 또는 모델이 이 기준선을 얼마나 초과하는지로 판단해야 한다."},
])
baseline_check.to_csv(os.path.join(OUT_TABLES, "target_baseline_check.csv"), index=False, encoding="utf-8-sig")
log("  target_baseline_check.csv 저장 완료")

# ─────────────────────────────────────────────
# 전처리 데이터 저장 (is_valid_race 플래그 포함)
# ─────────────────────────────────────────────
processed_path = os.path.join(
    os.path.dirname(OUT_TABLES), "..", "data", "processed", "race_data_preprocessed.csv"
)
processed_path = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "race_data_preprocessed.csv"
))
df.to_csv(processed_path, index=False, encoding="utf-8-sig")
log(f"  전처리 데이터 저장: {processed_path}")

print("\n[완료] 01_data_validation.py 실행 완료")
print(f"  - data_basic_summary.csv")
print(f"  - missing_value_summary.csv")
print(f"  - column_type_summary.csv")
print(f"  - race_integrity_check.csv ({n_invalid}건 문제 경주)")
print(f"  - target_baseline_check.csv (기준선: {arithmetic_base*100:.2f}%)")
