# -*- coding: utf-8 -*-
"""
utils.py - 공통 유틸리티 함수
"""

import logging
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from datetime import datetime

# ── 한글 폰트 설정 ─────────────────────────────────────────────────────────
def setup_korean_font():
    """matplotlib 한글 깨짐 방지 설정"""
    font_candidates = [
        "Malgun Gothic", "NanumGothic", "AppleGothic",
        "NanumBarunGothic", "Gulim", "Dotum",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in font_candidates:
        if font in available:
            matplotlib.rc("font", family=font)
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

setup_korean_font()

# ── 로거 설정 ──────────────────────────────────────────────────────────────
def get_logger(name: str, log_dir: Path = None) -> logging.Logger:
    """로거 생성. log_dir 지정 시 파일에도 기록"""
    logger = logging.getLogger(name)
    if logger.handlers:           # 중복 핸들러 방지
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    # 콘솔 핸들러
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # 파일 핸들러
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ── JSON 저장 ──────────────────────────────────────────────────────────────
def save_json(obj, path: Path):
    """딕셔너리/리스트를 JSON 파일로 저장"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── 차트 저장 헬퍼 ─────────────────────────────────────────────────────────
def save_fig(fig, path: Path, dpi: int = 150):
    """figure를 파일로 저장하고 닫기"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── 시간순 분할 ────────────────────────────────────────────────────────────
def time_split(df: pd.DataFrame,
               date_col: str,
               train_end: int,
               valid_end: int):
    """
    rcDate(정수 yyyymmdd) 기반 시간순 분할
    Returns: (train_df, valid_df, test_df)
    """
    train = df[df[date_col] <= train_end].copy()
    valid = df[(df[date_col] > train_end) & (df[date_col] <= valid_end)].copy()
    test  = df[df[date_col] > valid_end].copy()
    return train, valid, test


# ── race 내 정규화 ─────────────────────────────────────────────────────────
def normalize_within_race(df: pd.DataFrame,
                          score_col: str,
                          race_col: str = "race_id",
                          out_col: str = None) -> pd.DataFrame:
    """
    race_id 그룹 내에서 score_col을 [0,1]로 min-max 정규화
    동점 방지를 위해 약간의 epsilon 추가
    """
    out_col = out_col or f"{score_col}_norm"
    def _norm(x):
        rng = x.max() - x.min()
        if rng == 0:
            return pd.Series(np.ones(len(x)) / len(x), index=x.index)
        return (x - x.min()) / rng
    df[out_col] = df.groupby(race_col)[score_col].transform(_norm)
    return df


# ── 순위 계산 (race 내) ────────────────────────────────────────────────────
def rank_within_race(df: pd.DataFrame,
                     score_col: str,
                     race_col: str = "race_id",
                     ascending: bool = False) -> pd.Series:
    """
    race_id 그룹 내에서 score_col 기준 순위 계산
    ascending=False -> 높은 값이 1위
    """
    return df.groupby(race_col)[score_col].rank(
        method="min", ascending=ascending
    ).astype(int)


# ── 윈도우 성과 계산 헬퍼 ──────────────────────────────────────────────────
def rolling_win_rate(series: pd.Series, n: int) -> pd.Series:
    """최근 n전 승률 (shift 적용 전 호출 금지 - FE에서 shift 후 사용)"""
    return series.rolling(n, min_periods=1).mean()


# ── 피처 리스트 검증 ───────────────────────────────────────────────────────
def validate_feature_list(df: pd.DataFrame, features: list, logger=None) -> list:
    """
    DataFrame에 실제로 존재하는 피처만 필터링하여 반환
    누락된 피처는 로그에 기록
    """
    existing = [f for f in features if f in df.columns]
    missing  = [f for f in features if f not in df.columns]
    if missing and logger:
        logger.warning(f"피처 {len(missing)}개 누락 (DataFrame에 없음): {missing}")
    return existing


# ── Spearman 상관 ─────────────────────────────────────────────────────────
def spearman_by_race(df: pd.DataFrame,
                     pred_col: str,
                     true_col: str,
                     race_col: str = "race_id") -> float:
    """경주별 Spearman 순위상관의 평균"""
    from scipy.stats import spearmanr
    corrs = []
    for _, grp in df.groupby(race_col):
        if len(grp) < 2:
            continue
        corr, _ = spearmanr(grp[pred_col], grp[true_col])
        if not np.isnan(corr):
            corrs.append(corr)
    return float(np.mean(corrs)) if corrs else np.nan


# ── NDCG 계산 ─────────────────────────────────────────────────────────────
def ndcg_at_k(df: pd.DataFrame,
              score_col: str,
              true_rank_col: str,
              k: int,
              race_col: str = "race_id") -> float:
    """
    경주별 NDCG@k의 평균
    true_rank_col: 실제 순위 (1이 최고)
    score_col: 예측 점수 (높을수록 좋음)
    """
    results = []
    for _, grp in df.groupby(race_col):
        grp_sorted = grp.nlargest(k, score_col)
        # relevance = max(0, field_size - true_rank + 1) 방식
        rels = grp_sorted[true_rank_col].apply(
            lambda r: max(0, len(grp) - int(r) + 1)
        ).values

        dcg  = sum(r / np.log2(i + 2) for i, r in enumerate(rels))
        ideal_rels = sorted(
            [max(0, len(grp) - int(r) + 1) for r in grp[true_rank_col]],
            reverse=True
        )[:k]
        idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_rels))
        if idcg > 0:
            results.append(dcg / idcg)
    return float(np.mean(results)) if results else np.nan
