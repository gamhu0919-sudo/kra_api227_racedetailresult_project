# -*- coding: utf-8 -*-
"""
inference.py - 경주별 예측 추론 파이프라인
Streamlit 앱에서 호출하는 핵심 추론 함수 모음
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple

from config import DATA_PROCESSED, MODELS_DIR, ARTIFACTS_DIR, LOGS_DIR
from utils import (get_logger, load_json, rank_within_race,
                   normalize_within_race, validate_feature_list)

logger = get_logger("inference", LOGS_DIR)


# ── 전체 예측 결과 캐시 (앱 시동 시 1회 로드) ─────────────────────────────
_FINAL_PREDS_CACHE = {}


def load_predictions(mode: str = "no_odds") -> Optional[pd.DataFrame]:
    """
    사전 계산된 최종 예측 결과를 로드한다.
    mode: "no_odds" | "with_odds"
    """
    global _FINAL_PREDS_CACHE
    if mode in _FINAL_PREDS_CACHE:
        return _FINAL_PREDS_CACHE[mode]

    path = DATA_PROCESSED / f"final_predictions_{mode}.csv"
    fallback = DATA_PROCESSED / "final_predictions.csv"

    target = path if path.exists() else (fallback if fallback.exists() else None)
    if target is None:
        logger.warning(f"예측 결과 파일 없음: {path}")
        return None

    df = pd.read_csv(target, encoding="utf-8-sig", low_memory=False)
    df["rcDate"] = df["rcDate"].astype(str)
    _FINAL_PREDS_CACHE[mode] = df
    return df


def get_available_dates(mode: str = "no_odds") -> list:
    """사용 가능한 경주일 목록 반환 (최신순)"""
    df = load_predictions(mode)
    if df is None:
        return []
    dates = sorted(df["rcDate"].unique(), reverse=True)
    return dates


def get_available_races(race_date: str, mode: str = "no_odds") -> list:
    """특정 날짜의 경주번호 목록 반환"""
    df = load_predictions(mode)
    if df is None:
        return []
    day_df = df[df["rcDate"] == str(race_date)]
    return sorted(day_df["rcNo"].unique().tolist())


def get_race_prediction(race_date: str,
                         race_no: int,
                         mode: str = "no_odds") -> Optional[pd.DataFrame]:
    """
    특정 경주의 예측 결과 반환
    Returns: DataFrame (출전마별 예측 정보)
    """
    df = load_predictions(mode)
    if df is None:
        return None

    sub = df[(df["rcDate"] == str(race_date)) & (df["rcNo"] == int(race_no))].copy()
    if sub.empty:
        return None

    # 필수 컬럼 확보
    if "pred_rank" not in sub.columns and "final_score" in sub.columns:
        sub["pred_rank"] = rank_within_race(sub, "final_score", ascending=False)

    return sub.sort_values("pred_rank")


def get_reason_labels(row: pd.Series) -> list:
    """
    예측 근거 상위 3개 텍스트 반환
    실제 피처값 기반 - 임의 생성 금지
    """
    reasons = []

    # 1. 최근 3전 평균 순위
    if "hr_recent3_avg_rank" in row.index and pd.notna(row["hr_recent3_avg_rank"]):
        val = row["hr_recent3_avg_rank"]
        if val <= 3:
            reasons.append(f"최근 3전 평균 순위 우수 ({val:.1f}위)")
        elif val <= 5:
            reasons.append(f"최근 3전 평균 순위 양호 ({val:.1f}위)")

    # 2. 최근 3전 Top3율
    if "hr_recent3_top3_rate" in row.index and pd.notna(row["hr_recent3_top3_rate"]):
        val = row["hr_recent3_top3_rate"]
        if val >= 0.6:
            reasons.append(f"최근 3전 Top3 입상률 높음 ({val*100:.0f}%)")

    # 3. 기수 성과
    if "jk_90d_win_rate" in row.index and pd.notna(row["jk_90d_win_rate"]):
        val = row["jk_90d_win_rate"]
        if val >= 0.15:
            reasons.append(f"최근 90일 기수 승률 우수 ({val*100:.1f}%)")

    # 4. 말-기수 조합
    if "hr_jk_cum_win_rate" in row.index and pd.notna(row["hr_jk_cum_win_rate"]):
        val = row["hr_jk_cum_win_rate"]
        if val >= 0.2:
            reasons.append(f"말-기수 조합 승률 양호 ({val*100:.1f}%)")

    # 5. 조교사 폼
    if "tr_30d_win_rate" in row.index and pd.notna(row["tr_30d_win_rate"]):
        val = row["tr_30d_win_rate"]
        if val >= 0.15:
            reasons.append(f"최근 30일 조교사 폼 우수 ({val*100:.1f}%)")

    # 6. 누적 win율
    if "hr_cum_win_rate" in row.index and pd.notna(row["hr_cum_win_rate"]):
        val = row["hr_cum_win_rate"]
        if val >= 0.2:
            reasons.append(f"누적 승률 높음 ({val*100:.1f}%)")

    # 7. 배당 (With-Odds 모드)
    if "winOdds" in row.index and pd.notna(row["winOdds"]):
        val = row["winOdds"]
        if val <= 3.0:
            reasons.append(f"시장 배당 우위 ({val:.1f}배)")

    # 8. 직전 순위
    if "hr_prev1_rank" in row.index and pd.notna(row["hr_prev1_rank"]):
        val = row["hr_prev1_rank"]
        if val <= 2:
            reasons.append(f"직전 경주 {int(val)}위 입상")

    # 최소 1개 보장, 최대 3개
    if not reasons:
        reasons.append("예측 근거 정보 부족 (데이터 부족)")
    return reasons[:3]


def format_prediction_table(sub_df: pd.DataFrame,
                              show_actual: bool = False) -> pd.DataFrame:
    """
    Streamlit 표시용 DataFrame 포맷팅
    """
    cols_map = {
        "chulNo":      "출전번호",
        "hrName":      "말명",
        "jkName":      "기수",
        "trName":      "조교사",
        "pred_rank":   "예상순위",
        "final_score": "종합Score",
        "win_prob":    "1위확률",
        "top3_prob":   "Top3확률",
    }
    if show_actual and "ord" in sub_df.columns:
        cols_map["ord"] = "실제순위"
    if "winOdds" in sub_df.columns:
        cols_map["winOdds"] = "배당(단승)"

    avail_cols = {k: v for k, v in cols_map.items() if k in sub_df.columns}
    display = sub_df[list(avail_cols.keys())].rename(columns=avail_cols).copy()

    # 소수점 포맷
    for col in ["종합Score", "1위확률", "Top3확률"]:
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "-"
            )

    # 근거 컬럼 추가
    reasons_list = []
    for _, row in sub_df.iterrows():
        r = get_reason_labels(row)
        reasons_list.append(" / ".join(r))
    display["예측 근거"] = reasons_list

    return display.reset_index(drop=True)


def get_data_summary() -> dict:
    """데이터 점검 화면용 요약 정보"""
    from config import MAIN_CSV, REPORTS_TABLES

    summary = {
        "데이터_기간": "2025-03-01 ~ 2026-03-29",
        "총_경주수": 1150,
        "총_출전레코드수": 12206,
        "경마장": "서울",
        "특수코드": "92(35건), 93(2건), 94(133건), 95(79건) 총 249건(2.04%)",
        "분할": "Train(~2025-12) / Valid(2026-01) / Test(2026-02~03)",
    }

    # 결측률 상위 (테이블 파일에서 읽기)
    miss_path = REPORTS_TABLES / "02_missing_rate.csv"
    if miss_path.exists():
        miss_df = pd.read_csv(miss_path, encoding="utf-8-sig")
        top_miss = miss_df[miss_df["missing_rate_pct"] > 0].nlargest(5, "missing_rate_pct")
        summary["결측률_상위5"] = top_miss[["column", "missing_rate_pct"]].to_dict("records")

    # 피처 수
    try:
        no_odds = load_json(ARTIFACTS_DIR / "feature_list_no_odds.json")
        summary["No_Odds_피처수"] = len(no_odds.get("features", []))
    except Exception:
        summary["No_Odds_피처수"] = "N/A"

    try:
        with_odds = load_json(ARTIFACTS_DIR / "feature_list_with_odds.json")
        summary["With_Odds_피처수"] = len(with_odds.get("features", []))
    except Exception:
        summary["With_Odds_피처수"] = "N/A"

    return summary
