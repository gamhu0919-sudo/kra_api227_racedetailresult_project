# -*- coding: utf-8 -*-
"""
train_baselines.py - 베이스라인 모델 구축
실행: python src/train_baselines.py

베이스라인:
  A-1. With-Odds 규칙 기반 (winOdds 기준)
  A-2. No-Odds 규칙 기반 (최근 성적 점수)
  B.   약한 ML (Logistic Regression, RandomForest)

산출물:
  reports/GPT/outputs/tables/10_baseline_results.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from config import DATA_PROCESSED, REPORTS_TABLES, MODELS_DIR, LOGS_DIR, SEED
from utils import get_logger, ndcg_at_k, spearman_by_race, rank_within_race

logger = get_logger("train_baselines", LOGS_DIR)


def main():
    logger.info("=" * 60)
    logger.info("베이스라인 모델 구축 시작")
    logger.info("=" * 60)

    # 데이터 로드
    train = load_split("train")
    valid = load_split("valid")
    test  = load_split("test")

    if train is None:
        logger.error("train_fe.csv 없음. feature_engineering.py 먼저 실행하세요.")
        return

    results = []

    # ── A-1. With-Odds 규칙 기반 ──────────────────────────────────────────
    res_a1 = baseline_with_odds(test)
    if res_a1:
        results.append({"model": "A1_Odds_Rule", **res_a1})
        logger.info(f"A-1 결과: {res_a1}")

    # ── A-2. No-Odds 규칙 기반 ────────────────────────────────────────────
    res_a2 = baseline_no_odds(test)
    if res_a2:
        results.append({"model": "A2_NoOdds_Rule", **res_a2})
        logger.info(f"A-2 결과: {res_a2}")

    # ── B. 약한 ML 모델 ───────────────────────────────────────────────────
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    # No-Odds 피처 목록 로드
    from utils import load_json
    from config import ARTIFACTS_DIR

    try:
        feat_info = load_json(ARTIFACTS_DIR / "feature_list_no_odds.json")
        features  = [f for f in feat_info["features"] if f in train.columns]
    except Exception:
        # 기본 피처 목록
        features = [c for c in train.columns
                    if c not in ["race_id","entry_id","hrNo","jkNo","trNo","owNo",
                                 "hrName","jkName","trName","owName","rcDate","rcDate_dt",
                                 "rcNo","ord","target_win","target_top3",
                                 "winOdds","plcOdds","implied_rank_from_win_odds"]]

    num_features = [f for f in features if train[f].dtype in [np.float64, np.int64, float, int]]

    X_train = train[num_features].values
    y_win_train  = train["target_win"].values
    y_top3_train = train["target_top3"].values

    X_test = test[num_features].values
    y_win_test  = test["target_win"].values
    y_top3_test = test["target_top3"].values

    # 전처리
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_imp = imp.fit_transform(X_train)
    X_train_sc  = scaler.fit_transform(X_train_imp)
    X_test_imp  = imp.transform(X_test)
    X_test_sc   = scaler.transform(X_test_imp)

    # LR (Win)
    lr_win = LogisticRegression(max_iter=500, random_state=SEED, C=0.1)
    lr_win.fit(X_train_sc, y_win_train)

    # LR (Top3)
    lr_top3 = LogisticRegression(max_iter=500, random_state=SEED, C=0.1)
    lr_top3.fit(X_train_sc, y_top3_train)

    test_lr = test.copy()
    test_lr["lr_win_prob"]  = lr_win.predict_proba(X_test_sc)[:, 1]
    test_lr["lr_top3_prob"] = lr_top3.predict_proba(X_test_sc)[:, 1]
    test_lr["lr_score"]     = (test_lr["lr_win_prob"] + test_lr["lr_top3_prob"]) / 2
    test_lr["lr_pred_rank"] = rank_within_race(test_lr, "lr_score", ascending=False)

    res_lr = evaluate_baseline(test_lr, score_col="lr_score",
                                pred_rank_col="lr_pred_rank",
                                win_prob_col="lr_win_prob")
    results.append({"model": "B1_LR_NoOdds", **res_lr})
    logger.info(f"LR NoOdds 결과: {res_lr}")

    # RF (Win)
    rf_win = RandomForestClassifier(n_estimators=100, max_depth=6,
                                    random_state=SEED, n_jobs=-1)
    rf_win.fit(X_train_imp, y_win_train)

    rf_top3 = RandomForestClassifier(n_estimators=100, max_depth=6,
                                     random_state=SEED, n_jobs=-1)
    rf_top3.fit(X_train_imp, y_top3_train)

    test_rf = test.copy()
    test_rf["rf_win_prob"]  = rf_win.predict_proba(X_test_imp)[:, 1]
    test_rf["rf_top3_prob"] = rf_top3.predict_proba(X_test_imp)[:, 1]
    test_rf["rf_score"]     = (test_rf["rf_win_prob"] + test_rf["rf_top3_prob"]) / 2
    test_rf["rf_pred_rank"] = rank_within_race(test_rf, "rf_score", ascending=False)

    res_rf = evaluate_baseline(test_rf, score_col="rf_score",
                                pred_rank_col="rf_pred_rank",
                                win_prob_col="rf_win_prob")
    results.append({"model": "B2_RF_NoOdds", **res_rf})
    logger.info(f"RF NoOdds 결과: {res_rf}")

    # 모델 저장
    with open(MODELS_DIR / "baseline_lr_win.pkl", "wb")  as f: pickle.dump(lr_win, f)
    with open(MODELS_DIR / "baseline_lr_top3.pkl", "wb") as f: pickle.dump(lr_top3, f)
    with open(MODELS_DIR / "baseline_rf_win.pkl", "wb")  as f: pickle.dump(rf_win, f)
    with open(MODELS_DIR / "baseline_rf_top3.pkl", "wb") as f: pickle.dump(rf_top3, f)
    with open(MODELS_DIR / "baseline_imputer.pkl", "wb") as f: pickle.dump(imp, f)
    with open(MODELS_DIR / "baseline_scaler.pkl",  "wb") as f: pickle.dump(scaler, f)
    logger.info("베이스라인 모델 저장 완료")

    # 결과 저장
    result_df = pd.DataFrame(results)
    result_df["leakage_safe"] = "Y"
    result_df.to_csv(REPORTS_TABLES / "10_baseline_results.csv",
                     index=False, encoding="utf-8-sig")
    logger.info(f"\n베이스라인 결과 저장: {REPORTS_TABLES / '10_baseline_results.csv'}")
    logger.info("\n베이스라인 성능 요약:")
    logger.info(result_df.to_string(index=False))

    logger.info("=" * 60)
    logger.info("베이스라인 구축 완료")
    return result_df


def load_split(name: str):
    path = DATA_PROCESSED / f"{name}_fe.csv"
    if not path.exists():
        logger.warning(f"{path} 없음")
        return None
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df["ord"] = pd.to_numeric(df["ord"], errors="coerce")
    if "target_win" not in df.columns:
        df["target_win"]  = (df["ord"] == 1).astype(int)
    if "target_top3" not in df.columns:
        df["target_top3"] = (df["ord"] <= 3).astype(int)
    logger.info(f"{name}: {df.shape}")
    return df


def baseline_with_odds(df: pd.DataFrame) -> dict:
    """A-1: winOdds 기준 규칙 (낮을수록 1위 후보)"""
    if "winOdds" not in df.columns or df["winOdds"].isnull().all():
        logger.warning("winOdds 없음 - A-1 건너뜀")
        return None

    df2 = df.copy()
    df2["odds_score"] = -df2["winOdds"].fillna(df2["winOdds"].max() + 1)
    df2["odds_pred_rank"] = rank_within_race(df2, "odds_score", ascending=False)

    return evaluate_baseline(df2, score_col="odds_score",
                              pred_rank_col="odds_pred_rank",
                              win_prob_col=None)


def baseline_no_odds(df: pd.DataFrame) -> dict:
    """A-2: 최근 성적 기반 규칙 점수"""
    available = {c: c in df.columns for c in
                 ["hr_recent3_avg_rank", "hr_recent3_top3_rate", "hr_cum_win_rate",
                  "jk_cum_win_rate", "tr_cum_win_rate"]}
    logger.info(f"  NoOdds 규칙 피처 가용 여부: {available}")

    df2 = df.copy()
    score = pd.Series(0.0, index=df2.index)

    if "hr_recent3_avg_rank" in df2.columns:
        rank_norm = df2["hr_recent3_avg_rank"].fillna(df2["hr_recent3_avg_rank"].median())
        score -= rank_norm * 1.5         # 낮은 순위가 좋음

    if "hr_recent3_top3_rate" in df2.columns:
        score += df2["hr_recent3_top3_rate"].fillna(0) * 2.0

    if "hr_cum_win_rate" in df2.columns:
        score += df2["hr_cum_win_rate"].fillna(0) * 1.5

    if "jk_cum_win_rate" in df2.columns:
        score += df2["jk_cum_win_rate"].fillna(0) * 1.0

    if "tr_cum_win_rate" in df2.columns:
        score += df2["tr_cum_win_rate"].fillna(0) * 0.5

    df2["no_odds_score"] = score
    df2["no_odds_pred_rank"] = rank_within_race(df2, "no_odds_score", ascending=False)

    return evaluate_baseline(df2, score_col="no_odds_score",
                              pred_rank_col="no_odds_pred_rank",
                              win_prob_col=None)


def evaluate_baseline(df: pd.DataFrame,
                       score_col: str,
                       pred_rank_col: str,
                       win_prob_col=None) -> dict:
    """공통 베이스라인 평가"""
    df2 = df.dropna(subset=["ord"]).copy()
    df2["_true_rank"] = df2["ord"]

    # Winner Hit Rate (pred_rank==1 중 실제 1위 비율)
    top1 = df2[df2[pred_rank_col] == 1]
    win_hr = (top1["_true_rank"] == 1).mean() if len(top1) else 0.0

    # Top3 Hit Rate (pred_rank<=3 중 실제 top3 비율)
    top3_pred = df2[df2[pred_rank_col] <= 3]
    top3_hr = (top3_pred["_true_rank"] <= 3).mean() if len(top3_pred) else 0.0

    # NDCG
    ndcg3 = ndcg_at_k(df2, score_col, "_true_rank", k=3)
    ndcg5 = ndcg_at_k(df2, score_col, "_true_rank", k=5)

    # Spearman
    spear = spearman_by_race(df2, pred_rank_col, "_true_rank")

    # Rank MAE
    rank_mae = abs(df2[pred_rank_col] - df2["_true_rank"]).mean()

    result = {
        "winner_hit_rate":  round(win_hr, 4),
        "top3_hit_rate":    round(top3_hr, 4),
        "ndcg_3":           round(ndcg3, 4) if not np.isnan(ndcg3) else None,
        "ndcg_5":           round(ndcg5, 4) if not np.isnan(ndcg5) else None,
        "spearman":         round(spear, 4) if not np.isnan(spear) else None,
        "rank_mae":         round(rank_mae, 4),
        "uses_odds":        "Y" if win_prob_col == "odds_score" or "odds" in score_col else "N",
    }

    # Brier / LogLoss (win 확률 있을 때만)
    if win_prob_col and win_prob_col in df2.columns:
        from sklearn.metrics import brier_score_loss, log_loss
        y_true = df2["target_win"].values
        y_prob = df2[win_prob_col].values
        result["brier_win"] = round(brier_score_loss(y_true, y_prob), 4)
        result["logloss_win"] = round(log_loss(y_true, np.clip(y_prob, 1e-7, 1-1e-7)), 4)

    return result


if __name__ == "__main__":
    main()
