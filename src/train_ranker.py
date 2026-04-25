# -*- coding: utf-8 -*-
"""
train_ranker.py - CatBoostRanker / LightGBMRanker 학습
실행: python src/train_ranker.py

우선순위: CatBoostRanker → LightGBMRanker → XGBoost fallback

산출물:
  models/ranker_no_odds.*
  models/ranker_with_odds.*
  artifacts/ranker_no_odds_features.json
  artifacts/ranker_with_odds_features.json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

from config import DATA_PROCESSED, MODELS_DIR, LOGS_DIR, ARTIFACTS_DIR, SEED
from utils import (get_logger, load_json, save_json,
                   ndcg_at_k, spearman_by_race, rank_within_race)

logger = get_logger("train_ranker", LOGS_DIR)


def main():
    logger.info("=" * 60)
    logger.info("Ranker 모델 학습 시작")
    logger.info("=" * 60)

    train = load_split("train")
    valid = load_split("valid")
    test  = load_split("test")

    if train is None:
        logger.error("train_fe.csv 없음. feature_engineering.py 먼저 실행하세요.")
        return

    # 피처 목록 로드
    no_odds_feats   = get_features("no_odds", train)
    with_odds_feats = get_features("with_odds", train)

    logger.info(f"No-Odds 피처: {len(no_odds_feats)}개")
    logger.info(f"With-Odds 피처: {len(with_odds_feats)}개")

    results = {}

    # ── No-Odds Ranker ─────────────────────────────────────────────────────
    logger.info("\n[No-Odds Ranker 학습]")
    ranker_no, preds_no = train_ranker(
        train, valid, test,
        features=no_odds_feats,
        model_name="ranker_no_odds",
    )
    if preds_no is not None:
        results["NoOdds_Ranker"] = evaluate_ranker(preds_no, "rank_score")

    # ── With-Odds Ranker ───────────────────────────────────────────────────
    logger.info("\n[With-Odds Ranker 학습]")
    ranker_with, preds_with = train_ranker(
        train, valid, test,
        features=with_odds_feats,
        model_name="ranker_with_odds",
    )
    if preds_with is not None:
        results["WithOdds_Ranker"] = evaluate_ranker(preds_with, "rank_score")

    # 결과 저장
    from config import REPORTS_TABLES
    if results:
        res_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
        res_df.to_csv(REPORTS_TABLES / "11_ranker_results.csv",
                      index=False, encoding="utf-8-sig")
        logger.info("\n랭커 성능 요약:")
        logger.info(res_df.to_string(index=False))

    # 예측 결과 저장
    if preds_no is not None:
        preds_no.to_csv(DATA_PROCESSED / "test_ranker_no_odds.csv",
                        index=False, encoding="utf-8-sig")
    if preds_with is not None:
        preds_with.to_csv(DATA_PROCESSED / "test_ranker_with_odds.csv",
                          index=False, encoding="utf-8-sig")

    logger.info("=" * 60)
    logger.info("Ranker 학습 완료")
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
    logger.info(f"{name}: {df.shape}")
    return df


def get_features(mode: str, df: pd.DataFrame) -> list:
    try:
        info = load_json(ARTIFACTS_DIR / f"feature_list_{mode}.json")
        feats = [f for f in info["features"] if f in df.columns]
        num_feats = [f for f in feats if pd.api.types.is_numeric_dtype(df[f])]
        return num_feats
    except Exception:
        # 기본값
        exclude = {"race_id","entry_id","hrNo","jkNo","trNo","owNo",
                   "hrName","jkName","trName","owName","rcDate","rcDate_dt",
                   "rcNo","ord","target_win","target_top3",
                   "winOdds","plcOdds","implied_rank_from_win_odds"}
        if mode == "with_odds":
            exclude -= {"winOdds","plcOdds","implied_rank_from_win_odds"}
        return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def get_group_sizes(df: pd.DataFrame) -> list:
    """race_id 기준 그룹 크기 배열 (Ranker 학습용)"""
    return df.groupby("race_id")["ord"].count().values.tolist()


def train_ranker(train, valid, test, features, model_name):
    """CatBoostRanker 우선, 실패 시 LightGBM, XGBoost 순으로 fallback"""

    # 결측 채우기 (Ranker는 NaN 허용 여부 모델별 상이)
    X_train = train[features].fillna(-999)
    y_train = train["ord"].values
    X_valid = valid[features].fillna(-999) if valid is not None else None
    y_valid = valid["ord"].values if valid is not None else None
    X_test  = test[features].fillna(-999)

    # CatBoost용 표준 타입 group_id (ArrowStringArray 회피)
    grp_tr_cb = train.sort_values("race_id")["race_id"].astype(str).tolist()
    if valid is not None:
        grp_vl_cb = valid.sort_values("race_id")["race_id"].astype(str).tolist()

    # ── 1. CatBoostRanker ─────────────────────────────────────────────────
    try:
        from catboost import CatBoost, Pool
        logger.info("  CatBoostRanker 시도...")

        train_pool = Pool(
            data=X_train, label=y_train,
            group_id=grp_tr_cb
        )
        valid_pool = Pool(
            data=X_valid, label=y_valid,
            group_id=grp_vl_cb
        ) if valid is not None else None

        params = {
            "loss_function": "YetiRank",
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "random_seed": SEED,
            "verbose": 100,
            "early_stopping_rounds": 50,
        }
        model = CatBoost(params)
        train_sorted = train.sort_values("race_id")
        valid_sorted = valid.sort_values("race_id") if valid is not None else None

        # Pool 재구성 (정렬된 순서)
        X_tr_s = train_sorted[features].fillna(-999)
        y_tr_s = train_sorted["ord"].values
        tr_pool = Pool(data=X_tr_s, label=y_tr_s,
                       group_id=train_sorted["race_id"].astype(str).tolist())

        if valid is not None:
            X_vl_s = valid_sorted[features].fillna(-999)
            y_vl_s = valid_sorted["ord"].values
            vl_pool = Pool(data=X_vl_s, label=y_vl_s,
                           group_id=valid_sorted["race_id"].astype(str).tolist())
            model.fit(tr_pool, eval_set=vl_pool)
        else:
            model.fit(tr_pool)

        # feature importance 저장
        fi = pd.DataFrame({
            "feature": features,
            "importance": model.get_feature_importance(),
        }).sort_values("importance", ascending=False)
        fi.to_csv(MODELS_DIR / f"{model_name}_feature_importance.csv",
                  index=False, encoding="utf-8-sig")

        model.save_model(str(MODELS_DIR / f"{model_name}.cbm"))
        save_json({"features": features, "model_type": "CatBoostRanker"},
                  ARTIFACTS_DIR / f"{model_name}_features.json")

        # 예측
        test_sorted = test.sort_values(["race_id", "chulNo"])
        X_te_s = test_sorted[features].fillna(-999)
        scores  = model.predict(X_te_s)
        test_sorted["rank_score"] = scores
        test_sorted["pred_rank"]  = rank_within_race(test_sorted, "rank_score",
                                                      ascending=False)
        logger.info(f"  CatBoostRanker 학습 완료: {model_name}.cbm")
        return model, test_sorted

    except Exception as e:
        logger.warning(f"  CatBoostRanker 실패: {e}")

    # ── 2. LightGBMRanker fallback ────────────────────────────────────────
    try:
        import lightgbm as lgb
        logger.info("  LightGBMRanker fallback 시도...")

        train_sorted = train.sort_values("race_id")
        valid_sorted = valid.sort_values("race_id") if valid is not None else None

        X_tr_s = train_sorted[features].fillna(-999).values
        y_tr_s = train_sorted["ord"].values
        grp_tr = train_sorted.groupby("race_id").size().values

        ds_train = lgb.Dataset(X_tr_s, label=y_tr_s, group=grp_tr,
                                feature_name=features)

        callbacks = [lgb.log_evaluation(100), lgb.early_stopping(50)]
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [3, 5],
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "seed": SEED,
            "verbose": -1,
        }

        if valid is not None:
            X_vl_s = valid_sorted[features].fillna(-999).values
            y_vl_s = valid_sorted["ord"].values
            grp_vl = valid_sorted.groupby("race_id").size().values
            ds_valid = lgb.Dataset(X_vl_s, label=y_vl_s, group=grp_vl,
                                    reference=ds_train)
            model = lgb.train(params, ds_train,
                              valid_sets=[ds_valid],
                              callbacks=callbacks)
        else:
            model = lgb.train(params, ds_train)

        # feature importance
        fi = pd.DataFrame({
            "feature": features,
            "importance": model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)
        fi.to_csv(MODELS_DIR / f"{model_name}_feature_importance.csv",
                  index=False, encoding="utf-8-sig")

        model.save_model(str(MODELS_DIR / f"{model_name}.lgb"))
        save_json({"features": features, "model_type": "LGBMRanker"},
                  ARTIFACTS_DIR / f"{model_name}_features.json")

        test_sorted = test.sort_values(["race_id", "chulNo"])
        X_te_s = test_sorted[features].fillna(-999).values
        scores = model.predict(X_te_s)
        test_sorted["rank_score"] = scores
        test_sorted["pred_rank"]  = rank_within_race(test_sorted, "rank_score",
                                                      ascending=False)
        logger.info(f"  LightGBMRanker 완료: {model_name}.lgb")
        return model, test_sorted

    except Exception as e:
        logger.warning(f"  LightGBMRanker 실패: {e}")

    # ── 3. XGBoost fallback ───────────────────────────────────────────────
    try:
        import xgboost as xgb
        logger.info("  XGBoost pairwise fallback 시도...")

        train_sorted = train.sort_values("race_id")
        X_tr = train_sorted[features].fillna(-999).values
        y_tr = train_sorted["ord"].values
        grp_tr = train_sorted.groupby("race_id").size().values

        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=features)
        dtrain.set_group(grp_tr)

        params = {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg@3",
            "eta": 0.05,
            "max_depth": 6,
            "seed": SEED,
        }
        model = xgb.train(params, dtrain, num_boost_round=300,
                          verbose_eval=100)

        model.save_model(str(MODELS_DIR / f"{model_name}.xgb"))
        save_json({"features": features, "model_type": "XGBRanker"},
                  ARTIFACTS_DIR / f"{model_name}_features.json")

        test_sorted = test.sort_values(["race_id", "chulNo"])
        X_te = test_sorted[features].fillna(-999).values
        dtest = xgb.DMatrix(X_te, feature_names=features)
        scores = model.predict(dtest)
        test_sorted["rank_score"] = scores
        test_sorted["pred_rank"]  = rank_within_race(test_sorted, "rank_score",
                                                      ascending=False)
        logger.info(f"  XGBoost fallback 완료")
        return model, test_sorted

    except Exception as e:
        logger.error(f"  모든 Ranker 실패: {e}")
        return None, None


def evaluate_ranker(df: pd.DataFrame, score_col: str) -> dict:
    df2 = df.dropna(subset=["ord"]).copy()

    top1_df   = df2[df2["pred_rank"] == 1]
    win_hr    = (top1_df["ord"] == 1).mean() if len(top1_df) else 0.0
    top3_p_df = df2[df2["pred_rank"] <= 3]
    top3_hr   = (top3_p_df["ord"] <= 3).mean() if len(top3_p_df) else 0.0

    ndcg3  = ndcg_at_k(df2, score_col, "ord", k=3)
    ndcg5  = ndcg_at_k(df2, score_col, "ord", k=5)
    spear  = spearman_by_race(df2, "pred_rank", "ord")
    r_mae  = abs(df2["pred_rank"] - df2["ord"]).mean()

    return {
        "winner_hit_rate": round(win_hr, 4),
        "top3_hit_rate":   round(top3_hr, 4),
        "ndcg_3":          round(ndcg3, 4) if not np.isnan(ndcg3) else None,
        "ndcg_5":          round(ndcg5, 4) if not np.isnan(ndcg5) else None,
        "spearman":        round(spear, 4) if not np.isnan(spear) else None,
        "rank_mae":        round(r_mae, 4),
    }


if __name__ == "__main__":
    main()
