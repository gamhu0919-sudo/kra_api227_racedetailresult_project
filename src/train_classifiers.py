# -*- coding: utf-8 -*-
"""
train_classifiers.py - Winner/Top3 분류기 학습
실행: python src/train_classifiers.py

산출물:
  models/classifier_win_no_odds.*
  models/classifier_top3_no_odds.*
  models/classifier_win_with_odds.*
  models/classifier_top3_with_odds.*
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from config import DATA_PROCESSED, MODELS_DIR, LOGS_DIR, ARTIFACTS_DIR, SEED
from utils import get_logger, load_json, save_json

logger = get_logger("train_classifiers", LOGS_DIR)


def main():
    logger.info("=" * 60)
    logger.info("분류기 모델 학습 시작")
    logger.info("=" * 60)

    train = load_split("train")
    valid = load_split("valid")
    test  = load_split("test")

    if train is None:
        logger.error("train_fe.csv 없음. feature_engineering.py 먼저 실행하세요.")
        return

    no_odds_feats   = get_features("no_odds",   train)
    with_odds_feats = get_features("with_odds", train)

    results = {}

    # ── No-Odds 분류기 ─────────────────────────────────────────────────────
    logger.info("\n[No-Odds Winner 분류기]")
    clf_win_no, pred_win_no = train_classifier(
        train, valid, test, no_odds_feats,
        target="target_win", model_name="classifier_win_no_odds"
    )
    if pred_win_no is not None:
        res = eval_classifier(pred_win_no, prob_col="win_prob", target="target_win")
        results["Win_NoOdds"] = res
        logger.info(f"  결과: {res}")

    logger.info("\n[No-Odds Top3 분류기]")
    clf_top3_no, pred_top3_no = train_classifier(
        train, valid, test, no_odds_feats,
        target="target_top3", model_name="classifier_top3_no_odds"
    )
    if pred_top3_no is not None:
        res = eval_classifier(pred_top3_no, prob_col="top3_prob", target="target_top3")
        results["Top3_NoOdds"] = res
        logger.info(f"  결과: {res}")

    # ── With-Odds 분류기 ───────────────────────────────────────────────────
    logger.info("\n[With-Odds Winner 분류기]")
    clf_win_with, pred_win_with = train_classifier(
        train, valid, test, with_odds_feats,
        target="target_win", model_name="classifier_win_with_odds"
    )
    if pred_win_with is not None:
        res = eval_classifier(pred_win_with, prob_col="win_prob", target="target_win")
        results["Win_WithOdds"] = res
        logger.info(f"  결과: {res}")

    logger.info("\n[With-Odds Top3 분류기]")
    clf_top3_with, pred_top3_with = train_classifier(
        train, valid, test, with_odds_feats,
        target="target_top3", model_name="classifier_top3_with_odds"
    )
    if pred_top3_with is not None:
        res = eval_classifier(pred_top3_with, prob_col="top3_prob", target="target_top3")
        results["Top3_WithOdds"] = res
        logger.info(f"  결과: {res}")

    # 결과 저장
    from config import REPORTS_TABLES
    res_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    res_df.to_csv(REPORTS_TABLES / "12_classifier_results.csv",
                  index=False, encoding="utf-8-sig")
    logger.info(f"\n분류기 결과 저장: 12_classifier_results.csv")
    logger.info(res_df.to_string(index=False))

    # 테스트 예측 저장
    for pred_df, name in [
        (pred_win_no,   "test_clf_win_no_odds"),
        (pred_top3_no,  "test_clf_top3_no_odds"),
        (pred_win_with, "test_clf_win_with_odds"),
        (pred_top3_with,"test_clf_top3_with_odds"),
    ]:
        if pred_df is not None:
            pred_df.to_csv(DATA_PROCESSED / f"{name}.csv",
                           index=False, encoding="utf-8-sig")

    logger.info("=" * 60)
    logger.info("분류기 학습 완료")
    return results


def load_split(name: str):
    path = DATA_PROCESSED / f"{name}_fe.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df["ord"] = pd.to_numeric(df["ord"], errors="coerce")
    if "target_win" not in df.columns:
        df["target_win"]  = (df["ord"] == 1).astype(int)
    if "target_top3" not in df.columns:
        df["target_top3"] = (df["ord"] <= 3).astype(int)
    return df


def get_features(mode: str, df: pd.DataFrame) -> list:
    try:
        info = load_json(ARTIFACTS_DIR / f"feature_list_{mode}.json")
        feats = [f for f in info["features"] if f in df.columns]
        num_feats = [f for f in feats if pd.api.types.is_numeric_dtype(df[f])]
        return num_feats
    except Exception:
        exclude = {"race_id","entry_id","hrNo","jkNo","trNo","owNo",
                   "hrName","jkName","trName","owName","rcDate","rcDate_dt",
                   "rcNo","ord","target_win","target_top3"}
        if mode == "no_odds":
            exclude |= {"winOdds","plcOdds","implied_rank_from_win_odds"}
        return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def train_classifier(train, valid, test, features, target, model_name):
    """CatBoostClassifier 학습 (실패 시 RandomForest fallback)"""

    X_train = train[features].fillna(-999)
    y_train = train[target].values
    X_valid = valid[features].fillna(-999) if valid is not None else None
    y_valid = valid[target].values if valid is not None else None
    X_test  = test[features].fillna(-999)

    prob_col = "win_prob" if "win" in target else "top3_prob"

    # ── CatBoostClassifier ────────────────────────────────────────────────
    try:
        from catboost import CatBoostClassifier, Pool
        logger.info(f"  CatBoostClassifier 시도: {model_name}")

        tr_pool = Pool(X_train, label=y_train)
        vl_pool = Pool(X_valid, label=y_valid) if valid is not None else None

        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        params = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": SEED,
            "verbose": 100,
            "early_stopping_rounds": 50,
            "scale_pos_weight": min(scale_pos, 10),
        }
        model = CatBoostClassifier(**params)
        if vl_pool:
            model.fit(tr_pool, eval_set=vl_pool)
        else:
            model.fit(tr_pool)

        # feature importance
        fi = pd.DataFrame({
            "feature": features,
            "importance": model.get_feature_importance(),
        }).sort_values("importance", ascending=False)
        fi.to_csv(MODELS_DIR / f"{model_name}_feature_importance.csv",
                  index=False, encoding="utf-8-sig")

        model.save_model(str(MODELS_DIR / f"{model_name}.cbm"))
        save_json({"features": features, "target": target,
                   "model_type": "CatBoostClassifier"},
                  ARTIFACTS_DIR / f"{model_name}_features.json")

        test_out = test.copy()
        test_out[prob_col] = model.predict_proba(X_test)[:, 1]
        logger.info(f"  CatBoostClassifier 완료")
        return model, test_out

    except Exception as e:
        logger.warning(f"  CatBoostClassifier 실패: {e}")

    # ── RandomForest fallback ─────────────────────────────────────────────
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        logger.info(f"  RandomForest fallback: {model_name}")

        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_train)
        X_te_imp = imp.transform(X_test)

        cls_weight = {0: 1, 1: min(scale_pos, 10)} if 'scale_pos' in dir() else "balanced"
        model = RandomForestClassifier(n_estimators=200, max_depth=8,
                                       random_state=SEED, n_jobs=-1,
                                       class_weight=cls_weight)
        model.fit(X_tr_imp, y_train)

        with open(MODELS_DIR / f"{model_name}.pkl", "wb") as f:
            pickle.dump({"model": model, "imputer": imp}, f)

        save_json({"features": features, "target": target,
                   "model_type": "RandomForestClassifier"},
                  ARTIFACTS_DIR / f"{model_name}_features.json")

        test_out = test.copy()
        test_out[prob_col] = model.predict_proba(X_te_imp)[:, 1]
        logger.info("  RandomForest fallback 완료")
        return model, test_out

    except Exception as e:
        logger.error(f"  모든 분류기 실패: {e}")
        return None, None


def eval_classifier(df: pd.DataFrame, prob_col: str, target: str) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    df2 = df.dropna(subset=[target, prob_col]).copy()
    y_true = df2[target].values
    y_prob = np.clip(df2[prob_col].values, 1e-7, 1 - 1e-7)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan

    return {
        "brier":   round(brier_score_loss(y_true, y_prob), 4),
        "logloss": round(log_loss(y_true, y_prob), 4),
        "auc":     round(auc, 4) if not np.isnan(auc) else None,
        "pos_rate": round(y_true.mean(), 4),
    }


if __name__ == "__main__":
    main()
