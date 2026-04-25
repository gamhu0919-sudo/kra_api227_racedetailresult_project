# -*- coding: utf-8 -*-
"""
data_audit.py - 데이터 감사 (Data Audit)
실행: python src/data_audit.py

산출물:
  reports/GPT/outputs/tables/01_column_classification.csv
  reports/GPT/outputs/tables/02_missing_rate.csv
  reports/GPT/outputs/tables/03_special_code_freq.csv
  reports/GPT/outputs/tables/04_race_stats.csv
  reports/GPT/outputs/charts/01_field_size_dist.png
  reports/GPT/outputs/charts/02_monthly_race_count.png
  reports/GPT/outputs/charts/03_special_code_dist.png
  reports/GPT/outputs/charts/04_ord_distribution.png
  reports/GPT/outputs/charts/05_age_dist.png
  reports/GPT/outputs/charts/06_weight_dist.png
  reports/GPT/outputs/charts/07_target_vs_feature.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    MAIN_CSV, REPORTS_TABLES, REPORTS_CHARTS, LOGS_DIR,
    LEAKAGE_COLS, ODDS_COLS, UNDEFINED_COLS, ANA_COLS_PATTERN,
    TRAIN_END_DATE, VALID_END_DATE,
)
from utils import get_logger, save_fig, setup_korean_font

setup_korean_font()
logger = get_logger("data_audit", LOGS_DIR)


# ── 메인 ───────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("데이터 감사 시작")
    logger.info("=" * 60)

    # ── 데이터 로드 ────────────────────────────────────────────────────────
    logger.info(f"데이터 로드: {MAIN_CSV}")
    df = pd.read_csv(MAIN_CSV, encoding="utf-8-sig", low_memory=False)
    logger.info(f"Shape: {df.shape}")

    # ── 4-1. 기본 구조 점검 ────────────────────────────────────────────────
    audit_basic_structure(df)

    # ── 4-2. 경주 단위 점검 ────────────────────────────────────────────────
    audit_race_level(df)

    # ── 4-3. 타깃 컬럼(ord) 점검 ──────────────────────────────────────────
    audit_target_column(df)

    # ── 4-4. 컬럼 위험도 분류 ─────────────────────────────────────────────
    audit_column_classification(df)

    # ── 차트 ──────────────────────────────────────────────────────────────
    plot_charts(df)

    logger.info("=" * 60)
    logger.info("데이터 감사 완료")
    logger.info("=" * 60)


# ── 4-1. 기본 구조 점검 ────────────────────────────────────────────────────
def audit_basic_structure(df):
    logger.info("\n[4-1] 기본 구조 점검")

    # dtype별 분류
    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols  = df.select_dtypes(include=["object"]).columns.tolist()
    other_cols = [c for c in df.columns if c not in num_cols + obj_cols]

    logger.info(f"  수치형: {len(num_cols)}개")
    logger.info(f"  문자형: {len(obj_cols)}개")
    logger.info(f"  기타: {len(other_cols)}개")

    # 결측률 표
    missing = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.values,
        "missing_count": df.isnull().sum().values,
        "missing_rate_pct": (df.isnull().sum() / len(df) * 100).round(2).values,
        "unique_count": df.nunique().values,
    }).sort_values("missing_rate_pct", ascending=False)

    path = REPORTS_TABLES / "02_missing_rate.csv"
    missing.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"  결측률 표 저장: {path}")

    # 100% 결측 컬럼
    full_missing = missing[missing["missing_rate_pct"] == 100]["column"].tolist()
    logger.info(f"  100% 결측 컬럼 수: {len(full_missing)}")

    # 중복 행
    dup_rows = df.duplicated().sum()
    logger.info(f"  중복 행 수: {dup_rows}")

    # 날짜 범위
    logger.info(f"  rcDate 범위: {df['rcDate'].min()} ~ {df['rcDate'].max()}")

    # race_id 구조
    logger.info(f"  race_id 유니크: {df['race_id'].nunique()}")
    logger.info(f"  entry_id 유니크: {df['entry_id'].nunique() if 'entry_id' in df.columns else 'N/A'}")

    return missing


# ── 4-2. 경주 단위 점검 ────────────────────────────────────────────────────
def audit_race_level(df):
    logger.info("\n[4-2] 경주 단위 점검")

    # 경주당 출전두수
    race_stats = df.groupby("race_id").agg(
        field_size=("chulNo", "count"),
        race_date=("rcDate", "first"),
        rcNo=("rcNo", "first"),
    ).reset_index()

    logger.info(f"  총 경주 수: {len(race_stats)}")
    logger.info(f"  경주당 출전두수 분포:\n{race_stats['field_size'].describe().to_string()}")

    # 월별 경주 수
    df2 = df.copy()
    df2["year_month"] = (df2["rcDate"] // 100).astype(str)
    monthly = df2.groupby("year_month").agg(
        race_count=("race_id", "nunique"),
        avg_field_size=("chulNo", "count"),
    ).reset_index()
    monthly["avg_field_size"] = (
        monthly["avg_field_size"] / monthly["race_count"]
    ).round(1)

    path = REPORTS_TABLES / "04_race_stats.csv"
    monthly.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"  월별 경주 통계 저장: {path}")

    # 엔티티 등장 빈도 상위
    for col, name in [("hrNo", "말"), ("jkNo", "기수"),
                       ("trNo", "조교사"), ("owNo", "마주")]:
        top = df[col].value_counts().head(10)
        logger.info(f"\n  상위 10 {name}:\n{top.to_string()}")

    return race_stats


# ── 4-3. 타깃 컬럼 점검 ───────────────────────────────────────────────────
def audit_target_column(df):
    logger.info("\n[4-3] 타깃 컬럼 ord 점검")

    ord_dist = df["ord"].value_counts().sort_index()
    logger.info(f"\n  ord 전체 분포:\n{ord_dist.to_string()}")

    # 특수코드
    special = df[df["ord"] >= 90]
    special_freq = special["ord"].value_counts().sort_index()
    logger.info(f"\n  특수코드(ord≥90) 빈도:\n{special_freq.to_string()}")
    logger.info(f"  특수코드 비중: {len(special)/len(df)*100:.2f}%")
    logger.info(f"  특수코드 제거 후 행 수: {len(df) - len(special)}")

    # 특수코드 표 저장
    special_table = pd.DataFrame({
        "ord_code": special_freq.index,
        "count": special_freq.values,
        "pct": (special_freq.values / len(df) * 100).round(3),
    })
    path = REPORTS_TABLES / "03_special_code_freq.csv"
    special_table.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"  특수코드 빈도표 저장: {path}")

    return special_freq


# ── 4-4. 컬럼 위험도 분류 ─────────────────────────────────────────────────
def audit_column_classification(df):
    logger.info("\n[4-4] 컬럼 위험도 분류")

    # ana 계열 컬럼
    ana_cols = [c for c in df.columns if ANA_COLS_PATTERN in c]

    rows = []
    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100

        # 그룹 분류
        if col in LEAKAGE_COLS or col in ana_cols or missing_pct == 100:
            group = "B_누설/제거"
            no_odds = "N"
            with_odds = "N"
            reason = (
                "경주 중/후 기록 or 결과 직접 파생" if col in LEAKAGE_COLS
                else "_ana계열 100%결측" if col in ana_cols
                else "100% 결측"
            )
        elif col in ODDS_COLS:
            group = "C_조건부(With-Odds)"
            no_odds = "N"
            with_odds = "Y"
            reason = "시장배당정보 - With-Odds 전용"
        elif col in UNDEFINED_COLS:
            group = "D_의미불명확"
            no_odds = "N"
            with_odds = "N"
            reason = "정의 확인 필요"
        else:
            group = "A_안전사전정보"
            no_odds = "Y"
            with_odds = "Y"
            reason = "출전 전 알 수 있는 정보"

        rows.append({
            "column_name": col,
            "dtype": str(df[col].dtype),
            "missing_rate_pct": round(missing_pct, 2),
            "group": group,
            "include_in_no_odds_model": no_odds,
            "include_in_with_odds_model": with_odds,
            "reason": reason,
            "remarks": "",
        })

    result = pd.DataFrame(rows)
    path = REPORTS_TABLES / "01_column_classification.csv"
    result.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"  컬럼 분류표 저장: {path}")

    # 요약
    for g, cnt in result["group"].value_counts().items():
        logger.info(f"    {g}: {cnt}개")

    return result


# ── 차트 ───────────────────────────────────────────────────────────────────
def plot_charts(df):
    logger.info("\n[차트 생성]")

    # 01. 경주당 출전두수 분포
    field_size = df.groupby("race_id")["chulNo"].count()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = field_size.value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="#4A90D9", edgecolor="white")
    ax.set_title("경주당 출전두수 분포", fontsize=14, fontweight="bold")
    ax.set_xlabel("출전두수")
    ax.set_ylabel("경주 수")
    for i, v in zip(counts.index, counts.values):
        ax.text(str(i), v + 0.5, str(v), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    save_fig(fig, REPORTS_CHARTS / "01_field_size_dist.png")
    logger.info("  01_field_size_dist.png 저장")

    # 02. 월별 경주 수
    df2 = df.copy()
    df2["year_month"] = (df2["rcDate"] // 100).astype(str)
    monthly_rc = df2.groupby("year_month")["race_id"].nunique()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(monthly_rc.index, monthly_rc.values, color="#2E86AB", edgecolor="white")
    ax.set_title("월별 경주 수", fontsize=14, fontweight="bold")
    ax.set_xlabel("연월")
    ax.set_ylabel("경주 수")
    ax.tick_params(axis="x", rotation=45)
    for x, v in zip(monthly_rc.index, monthly_rc.values):
        ax.text(x, v + 0.3, str(v), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    save_fig(fig, REPORTS_CHARTS / "02_monthly_race_count.png")
    logger.info("  02_monthly_race_count.png 저장")

    # 03. 특수코드 분포
    special = df[df["ord"] >= 90]["ord"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(special.index.astype(str), special.values,
           color=["#E74C3C", "#E67E22", "#F39C12", "#D35400"])
    ax.set_title("특수코드(ord≥90) 분포", fontsize=14, fontweight="bold")
    ax.set_xlabel("ord 코드")
    ax.set_ylabel("빈도")
    for i, v in zip(special.index.astype(str), special.values):
        ax.text(i, v + 0.3, str(v), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    save_fig(fig, REPORTS_CHARTS / "03_special_code_dist.png")
    logger.info("  03_special_code_dist.png 저장")

    # 04. ord 정상 분포 (1~11)
    normal = df[df["ord"] < 90]["ord"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(normal.index.astype(str), normal.values, color="#27AE60", edgecolor="white")
    ax.set_title("정상 완주 순위(ord) 분포", fontsize=14, fontweight="bold")
    ax.set_xlabel("순위")
    ax.set_ylabel("빈도")
    for i, v in zip(normal.index.astype(str), normal.values):
        ax.text(i, v + 2, str(v), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    save_fig(fig, REPORTS_CHARTS / "04_ord_distribution.png")
    logger.info("  04_ord_distribution.png 저장")

    # 05. 연령 분포
    if "age" in df.columns:
        age_counts = df["age"].value_counts().sort_index().dropna()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(age_counts.index.astype(str), age_counts.values,
               color="#8E44AD", edgecolor="white")
        ax.set_title("말 연령 분포", fontsize=14, fontweight="bold")
        ax.set_xlabel("연령")
        ax.set_ylabel("출전 수")
        plt.tight_layout()
        save_fig(fig, REPORTS_CHARTS / "05_age_dist.png")
        logger.info("  05_age_dist.png 저장")

    # 06. 체중 분포
    if "horse_weight_current" in df.columns:
        wt = df["horse_weight_current"].dropna()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(wt, bins=30, color="#16A085", edgecolor="white", alpha=0.85)
        ax.axvline(wt.mean(), color="red", linestyle="--",
                   label=f"평균 {wt.mean():.1f}kg")
        ax.set_title("출전마 체중 분포", fontsize=14, fontweight="bold")
        ax.set_xlabel("체중(kg)")
        ax.set_ylabel("빈도")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, REPORTS_CHARTS / "06_weight_dist.png")
        logger.info("  06_weight_dist.png 저장")

    # 07. 부담중량 분포 (타깃별)
    if "wgBudam" in df.columns:
        df_clean = df[df["ord"] < 90].copy()
        df_clean["is_winner"] = (df_clean["ord"] == 1).astype(int)
        win_wg  = df_clean[df_clean["is_winner"] == 1]["wgBudam"].dropna()
        lose_wg = df_clean[df_clean["is_winner"] == 0]["wgBudam"].dropna()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(lose_wg, bins=25, alpha=0.6, label="비1위", color="#95A5A6", edgecolor="white")
        ax.hist(win_wg,  bins=25, alpha=0.8, label="1위",   color="#E74C3C", edgecolor="white")
        ax.set_title("부담중량 분포 (1위 vs 비1위)", fontsize=13, fontweight="bold")
        ax.set_xlabel("부담중량(kg)")
        ax.set_ylabel("빈도")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, REPORTS_CHARTS / "07_target_vs_feature.png")
        logger.info("  07_target_vs_feature.png 저장")

    logger.info("  차트 생성 완료")


if __name__ == "__main__":
    main()
