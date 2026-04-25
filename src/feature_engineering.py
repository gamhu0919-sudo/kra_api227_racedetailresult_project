# -*- coding: utf-8 -*-
"""
feature_engineering.py - 시간누설 없는 피처 엔지니어링
실행: python src/feature_engineering.py

핵심 원칙: 날짜 오름차순 정렬 → shift(1) → rolling → split
모든 피처는 현재 경주 이전 데이터만 사용

산출물:
  data/processed/train_fe.csv
  data/processed/valid_fe.csv
  data/processed/test_fe.csv
  artifacts/feature_list_no_odds.json
  artifacts/feature_list_with_odds.json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from pathlib import Path

from config import (
    DATA_PROCESSED, ARTIFACTS_DIR, LOGS_DIR,
    TRAIN_END_DATE, VALID_END_DATE, SEED,
    ODDS_COLS,
)
from utils import get_logger, save_json, time_split, validate_feature_list

logger = get_logger("feature_engineering", LOGS_DIR)

# ── 피처 목록 (최종 결정) ──────────────────────────────────────────────────
STATIC_FEATURES = [
    "age", "sex_enc", "wgBudam", "wgJk",
    "chulNo", "chulNo_relative", "field_size",
    "tool_count", "has_tool",
    "hrRating",
    # 병합 후 컬럼 (없으면 자동 제외)
    "rcDist", "dist_group",
    "track_enc", "weather_enc",
    "rank_enc", "prizeCond_enc",
    # 결측 플래그
    "is_missing_horse_weight_current",
    "is_missing_wgBudam", "is_missing_wgJk",
]

HORSE_FEATURES = [
    "hr_prev1_rank",
    "hr_recent3_avg_rank", "hr_recent5_avg_rank",
    "hr_recent1_win_rate", "hr_recent3_win_rate", "hr_recent5_win_rate",
    "hr_recent1_top3_rate", "hr_recent3_top3_rate", "hr_recent5_top3_rate",
    "hr_recent3_avg_field_size",
    "hr_recent3_avg_weight_delta",
    "hr_recent3_avg_rest_days",
    "hr_last_rest_days",
    "hr_cum_starts", "hr_cum_win_rate", "hr_cum_top3_rate",
    "hr_rank_trend",
]

JOCKEY_FEATURES = [
    "jk_cum_starts", "jk_cum_win_rate", "jk_cum_top3_rate",
    "jk_30d_win_rate", "jk_30d_top3_rate",
    "jk_90d_win_rate", "jk_90d_top3_rate",
    "jk_recent_avg_rank",
]

TRAINER_FEATURES = [
    "tr_cum_starts", "tr_cum_win_rate", "tr_cum_top3_rate",
    "tr_30d_win_rate", "tr_30d_top3_rate",
    "tr_90d_win_rate", "tr_90d_top3_rate",
    "tr_recent_avg_rank",
]

OWNER_FEATURES = [
    "ow_cum_starts", "ow_cum_win_rate", "ow_cum_top3_rate",
    "ow_30d_win_rate", "ow_90d_win_rate",
    "ow_recent_avg_rank",
]

COMBO_FEATURES = [
    "hr_jk_cum_starts", "hr_jk_cum_win_rate", "hr_jk_cum_top3_rate",
    "hr_tr_cum_starts", "hr_tr_cum_win_rate", "hr_tr_cum_top3_rate",
]

NO_ODDS_FEATURES = (STATIC_FEATURES + HORSE_FEATURES + JOCKEY_FEATURES
                    + TRAINER_FEATURES + OWNER_FEATURES + COMBO_FEATURES)

WITH_ODDS_EXTRA = ["winOdds", "plcOdds", "implied_rank_from_win_odds"]
WITH_ODDS_FEATURES = NO_ODDS_FEATURES + WITH_ODDS_EXTRA

# 타깃/분할에 필요한 컬럼
META_COLS = ["race_id", "entry_id", "hrNo", "jkNo", "trNo", "owNo",
             "hrName", "jkName", "trName", "owName",
             "rcDate", "rcDate_dt", "rcNo", "chulNo",
             "ord", "target_win", "target_top3"]


def main():
    logger.info("=" * 60)
    logger.info("피처 엔지니어링 시작")
    logger.info("=" * 60)

    # df_clean 로드 (정상완주만)
    path = DATA_PROCESSED / "df_clean.csv"
    if not path.exists():
        logger.error(f"{path} 없음. preprocess.py 먼저 실행하세요.")
        return

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df["rcDate_dt"] = pd.to_datetime(df["rcDate"].astype(str), format="%Y%m%d", errors="coerce")
    df["ord"] = pd.to_numeric(df["ord"], errors="coerce")
    logger.info(f"로드: {df.shape}")

    # 날짜 오름차순 정렬 (핵심!)
    df = df.sort_values(["rcDate", "rcNo", "chulNo"]).reset_index(drop=True)

    # 타깃 생성
    df["target_win"]  = (df["ord"] == 1).astype(int)
    df["target_top3"] = (df["ord"] <= 3).astype(int)

    # ── 피처 생성 (전체 df 기준, shift로 누설 방지) ────────────────────────
    df = build_horse_features(df)
    df = build_jockey_features(df)
    df = build_trainer_features(df)
    df = build_owner_features(df)
    df = build_combo_features(df)

    logger.info(f"\n피처 생성 후 shape: {df.shape}")

    # ── 시간순 분할 ────────────────────────────────────────────────────────
    train, valid, test = time_split(df, "rcDate", TRAIN_END_DATE, VALID_END_DATE)
    logger.info(f"\nTrain: {train.shape} (rcDate <= {TRAIN_END_DATE})")
    logger.info(f"Valid: {valid.shape} ({TRAIN_END_DATE} < rcDate <= {VALID_END_DATE})")
    logger.info(f"Test:  {test.shape} (rcDate > {VALID_END_DATE})")

    # ── 최종 피처 목록 확정 ────────────────────────────────────────────────
    no_odds_final   = validate_feature_list(df, NO_ODDS_FEATURES, logger)
    with_odds_final = validate_feature_list(df, WITH_ODDS_FEATURES, logger)

    logger.info(f"\nNo-Odds 최종 피처 수: {len(no_odds_final)}")
    logger.info(f"With-Odds 최종 피처 수: {len(with_odds_final)}")

    # 피처 목록 저장
    save_json({"features": no_odds_final, "meta": META_COLS},
              ARTIFACTS_DIR / "feature_list_no_odds.json")
    save_json({"features": with_odds_final, "meta": META_COLS},
              ARTIFACTS_DIR / "feature_list_with_odds.json")
    logger.info("피처 목록 JSON 저장 완료")

    # ── 저장 (메타 + 피처 컬럼) ───────────────────────────────────────────
    keep_cols = list(set(META_COLS + with_odds_final))
    keep_cols = [c for c in keep_cols if c in df.columns]

    for split_df, name in [(train, "train"), (valid, "valid"), (test, "test")]:
        out_cols = [c for c in keep_cols if c in split_df.columns]
        split_df[out_cols].to_csv(
            DATA_PROCESSED / f"{name}_fe.csv",
            index=False, encoding="utf-8-sig"
        )
        logger.info(f"저장: {name}_fe.csv ({split_df.shape[0]}행, {len(out_cols)}컬럼)")

    # ── 피처 통계 ─────────────────────────────────────────────────────────
    report_feature_stats(train, no_odds_final, valid)

    logger.info("=" * 60)
    logger.info("피처 엔지니어링 완료")
    logger.info("=" * 60)
    return train, valid, test, no_odds_final, with_odds_final


# ── 말(hrNo) 기준 피처 ─────────────────────────────────────────────────────
def build_horse_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("\n[말 피처 생성]")

    # 누설 방지: 경주 단위로 그룹화하여 각 말의 "이전" 기록만 사용
    # 각 (hrNo, rcDate) 기준으로 shift(1)

    # 1. 경주 단위 집계 (말별 과거 성적 행)
    df = df.sort_values(["hrNo", "rcDate", "rcNo"]).reset_index(drop=True)

    g = df.groupby("hrNo")

    # 직전 순위
    df["hr_prev1_rank"] = g["ord"].shift(1)

    # 최근 N전 평균 순위 (shift(1) 후 rolling)
    for n in [3, 5]:
        df[f"hr_recent{n}_avg_rank"] = (
            g["ord"].shift(1).groupby(df["hrNo"]).transform(
                lambda x: x.rolling(n, min_periods=1).mean()
            )
        )

    # 최근 N전 win율
    win_shifted = g["target_win"].shift(1)
    top3_shifted = g["target_top3"].shift(1)

    for n in [1, 3, 5]:
        df[f"hr_recent{n}_win_rate"] = (
            win_shifted.groupby(df["hrNo"]).transform(
                lambda x: x.rolling(n, min_periods=1).mean()
            )
        )
        df[f"hr_recent{n}_top3_rate"] = (
            top3_shifted.groupby(df["hrNo"]).transform(
                lambda x: x.rolling(n, min_periods=1).mean()
            )
        )

    # 최근 3전 평균 출전두수
    df["hr_recent3_avg_field_size"] = (
        g["field_size"].shift(1).groupby(df["hrNo"]).transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
    )

    # 최근 3전 평균 체중변화
    if "horse_weight_delta" in df.columns:
        df["hr_recent3_avg_weight_delta"] = (
            g["horse_weight_delta"].shift(1).groupby(df["hrNo"]).transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
        )
    else:
        df["hr_recent3_avg_weight_delta"] = np.nan

    # 휴식일수 (이전 경주일로부터 경과일)
    df["rcDate_dt"] = pd.to_datetime(df["rcDate"].astype(str), format="%Y%m%d", errors="coerce")
    df["prev_rcDate_dt"] = g["rcDate_dt"].shift(1)
    df["hr_last_rest_days"] = (df["rcDate_dt"] - df["prev_rcDate_dt"]).dt.days

    # 최근 3전 평균 휴식일
    df["hr_recent3_avg_rest_days"] = (
        df.groupby("hrNo")["hr_last_rest_days"].shift(1).groupby(df["hrNo"]).transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
    )

    # 누적 출전 횟수
    df["hr_cum_starts"] = g["ord"].shift(1).groupby(df["hrNo"]).transform(
        lambda x: x.expanding().count()
    )

    # 누적 win율 / top3율
    df["hr_cum_win_rate"] = (
        win_shifted.groupby(df["hrNo"]).transform(lambda x: x.expanding().mean())
    )
    df["hr_cum_top3_rate"] = (
        top3_shifted.groupby(df["hrNo"]).transform(lambda x: x.expanding().mean())
    )

    # 순위 추세 (최근 3전 - 이전 3전 평균 순위 차이)
    ord_shifted = g["ord"].shift(1)
    recent3 = ord_shifted.groupby(df["hrNo"]).transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    prev3_shifted = recent3.groupby(df["hrNo"]).shift(3)
    df["hr_rank_trend"] = recent3 - prev3_shifted

    # 임시 컬럼 정리
    df = df.drop(columns=["prev_rcDate_dt"], errors="ignore")

    logger.info(f"  말 피처 생성 완료: {len(HORSE_FEATURES)}개 목표")
    return df


# ── 기수(jkNo) 기준 피처 ──────────────────────────────────────────────────
def build_jockey_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("[기수 피처 생성]")

    df = df.sort_values(["jkNo", "rcDate", "rcNo"]).reset_index(drop=True)
    g = df.groupby("jkNo")

    win_s  = g["target_win"].shift(1)
    top3_s = g["target_top3"].shift(1)
    ord_s  = g["ord"].shift(1)

    df["jk_cum_starts"]   = win_s.groupby(df["jkNo"]).transform(lambda x: x.expanding().count())
    df["jk_cum_win_rate"] = win_s.groupby(df["jkNo"]).transform(lambda x: x.expanding().mean())
    df["jk_cum_top3_rate"]= top3_s.groupby(df["jkNo"]).transform(lambda x: x.expanding().mean())
    df["jk_recent_avg_rank"] = ord_s.groupby(df["jkNo"]).transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    )

    # 30일 / 90일 성과 (날짜 기반)
    df = _rolling_date_stats(df, entity_col="jkNo",
                             win_col="target_win", top3_col="target_top3",
                             days_list=[30, 90], prefix="jk")

    logger.info("  기수 피처 완료")
    return df


# ── 조교사(trNo) 기준 피처 ────────────────────────────────────────────────
def build_trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("[조교사 피처 생성]")

    df = df.sort_values(["trNo", "rcDate", "rcNo"]).reset_index(drop=True)
    g = df.groupby("trNo")

    win_s  = g["target_win"].shift(1)
    top3_s = g["target_top3"].shift(1)
    ord_s  = g["ord"].shift(1)

    df["tr_cum_starts"]   = win_s.groupby(df["trNo"]).transform(lambda x: x.expanding().count())
    df["tr_cum_win_rate"] = win_s.groupby(df["trNo"]).transform(lambda x: x.expanding().mean())
    df["tr_cum_top3_rate"]= top3_s.groupby(df["trNo"]).transform(lambda x: x.expanding().mean())
    df["tr_recent_avg_rank"] = ord_s.groupby(df["trNo"]).transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    )

    df = _rolling_date_stats(df, entity_col="trNo",
                             win_col="target_win", top3_col="target_top3",
                             days_list=[30, 90], prefix="tr")
    logger.info("  조교사 피처 완료")
    return df


# ── 마주(owNo) 기준 피처 ──────────────────────────────────────────────────
def build_owner_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("[마주 피처 생성]")

    df = df.sort_values(["owNo", "rcDate", "rcNo"]).reset_index(drop=True)
    g = df.groupby("owNo")

    win_s  = g["target_win"].shift(1)
    top3_s = g["target_top3"].shift(1)
    ord_s  = g["ord"].shift(1)

    df["ow_cum_starts"]   = win_s.groupby(df["owNo"]).transform(lambda x: x.expanding().count())
    df["ow_cum_win_rate"] = win_s.groupby(df["owNo"]).transform(lambda x: x.expanding().mean())
    df["ow_cum_top3_rate"]= top3_s.groupby(df["owNo"]).transform(lambda x: x.expanding().mean())
    df["ow_recent_avg_rank"] = ord_s.groupby(df["owNo"]).transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    )

    df = _rolling_date_stats(df, entity_col="owNo",
                             win_col="target_win", top3_col="target_top3",
                             days_list=[30, 90], prefix="ow")
    logger.info("  마주 피처 완료")
    return df


# ── 조합(Interaction) 피처 ────────────────────────────────────────────────
def build_combo_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("[조합 피처 생성]")

    # 말-기수 조합
    df["hr_jk_key"] = df["hrNo"].astype(str) + "_" + df["jkNo"].astype(str)
    df = df.sort_values(["hr_jk_key", "rcDate", "rcNo"]).reset_index(drop=True)
    g = df.groupby("hr_jk_key")
    win_s  = g["target_win"].shift(1)
    top3_s = g["target_top3"].shift(1)
    df["hr_jk_cum_starts"]   = win_s.groupby(df["hr_jk_key"]).transform(lambda x: x.expanding().count())
    df["hr_jk_cum_win_rate"] = win_s.groupby(df["hr_jk_key"]).transform(lambda x: x.expanding().mean())
    df["hr_jk_cum_top3_rate"]= top3_s.groupby(df["hr_jk_key"]).transform(lambda x: x.expanding().mean())

    # 말-조교사 조합
    df["hr_tr_key"] = df["hrNo"].astype(str) + "_" + df["trNo"].astype(str)
    df = df.sort_values(["hr_tr_key", "rcDate", "rcNo"]).reset_index(drop=True)
    g2 = df.groupby("hr_tr_key")
    win_s2  = g2["target_win"].shift(1)
    top3_s2 = g2["target_top3"].shift(1)
    df["hr_tr_cum_starts"]   = win_s2.groupby(df["hr_tr_key"]).transform(lambda x: x.expanding().count())
    df["hr_tr_cum_win_rate"] = win_s2.groupby(df["hr_tr_key"]).transform(lambda x: x.expanding().mean())
    df["hr_tr_cum_top3_rate"]= top3_s2.groupby(df["hr_tr_key"]).transform(lambda x: x.expanding().mean())

    # 임시 키 컬럼 제거
    df = df.drop(columns=["hr_jk_key", "hr_tr_key"], errors="ignore")

    logger.info("  조합 피처 완료")
    return df


# ── 날짜 기반 롤링 통계 (30d/90d) - 벡터화 최적화 버전 ──────────────────
def _rolling_date_stats(df: pd.DataFrame,
                        entity_col: str,
                        win_col: str,
                        top3_col: str,
                        days_list: list,
                        prefix: str) -> pd.DataFrame:
    """
    특정 엔티티(entity_col)에 대해 경주일(rcDate_dt) 기준
    N일 이전 데이터만 사용한 win율 / top3율 계산
    merge 기반 벡터화로 iterrows 대비 수십배 빠름
    """
    df = df.sort_values([entity_col, "rcDate_dt", "rcNo"]).reset_index(drop=True)
    df["_rcDate_ts"] = df["rcDate_dt"].astype(np.int64)  # 나노초 단위

    for days in days_list:
        delta_ns = days * 86400 * 10**9  # days → 나노초

        # self-join: left = 현재, right = 과거 기록
        left  = df[[entity_col, "_rcDate_ts", "rcDate_dt"]].copy()
        right = df[[entity_col, "_rcDate_ts", win_col, top3_col]].copy()
        right = right.rename(columns={
            "_rcDate_ts": "_ts_past",
            win_col:  "_win_past",
            top3_col: "_top3_past",
        })

        merged = left.merge(right, on=entity_col, how="left")
        # 조건: past < current AND past >= current - delta
        merged = merged[
            (merged["_ts_past"] < merged["_rcDate_ts"]) &
            (merged["_ts_past"] >= merged["_rcDate_ts"] - delta_ns)
        ]

        agg = merged.groupby(merged.index.name or "index" if False else
                              # index 기반 집계를 위해 임시 컬럼 사용
                              entity_col)

        # 인덱스 보존을 위해 reset 후 집계
        merged2 = left.reset_index().merge(right, on=entity_col, how="left")
        merged2 = merged2[
            (merged2["_ts_past"] < merged2["_rcDate_ts"]) &
            (merged2["_ts_past"] >= merged2["_rcDate_ts"] - delta_ns)
        ]
        agg2 = merged2.groupby("index").agg(
            win_rate=("_win_past", "mean"),
            top3_rate=("_top3_past", "mean"),
        )

        df[f"{prefix}_{days}d_win_rate"]  = agg2["win_rate"]
        df[f"{prefix}_{days}d_top3_rate"] = agg2["top3_rate"]
        logger.info(f"  {prefix}_{days}d 계산 완료")

    df = df.drop(columns=["_rcDate_ts"], errors="ignore")
    return df


# ── 피처 통계 보고 ─────────────────────────────────────────────────────────
def report_feature_stats(train_df, features, valid_df):
    logger.info("\n[피처 통계]")
    avail = [f for f in features if f in train_df.columns]
    missing_rates = {
        f: round(train_df[f].isnull().mean() * 100, 2) for f in avail
    }
    high_missing = {k: v for k, v in missing_rates.items() if v > 30}
    if high_missing:
        logger.warning(f"  Train에서 결측률 30% 초과 피처: {high_missing}")

    # 피처 통계표 저장
    from config import REPORTS_TABLES
    def safe_mean(df, col):
        if col not in df.columns:
            return np.nan
        if pd.api.types.is_numeric_dtype(df[col]):
            return df[col].mean()
        return np.nan

    def safe_std(df, col):
        if col not in df.columns:
            return np.nan
        if pd.api.types.is_numeric_dtype(df[col]):
            return df[col].std()
        return np.nan

    stats_df = pd.DataFrame([
        {"feature": f, "train_missing_pct": missing_rates.get(f, 0),
         "train_mean": safe_mean(train_df, f),
         "train_std":  safe_std(train_df, f)}
        for f in features
    ])
    stats_df.to_csv(REPORTS_TABLES / "09_feature_stats.csv",
                    index=False, encoding="utf-8-sig")
    logger.info(f"  피처 통계표 저장: 09_feature_stats.csv")


if __name__ == "__main__":
    main()
