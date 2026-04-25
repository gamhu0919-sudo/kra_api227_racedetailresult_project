# -*- coding: utf-8 -*-
"""
calibration.py - 확률 보정 (Platt Scaling / Isotonic Regression)
실행: python src/calibration.py

산출물:
  models/calibrator_win_no_odds.pkl
  models/calibrator_win_with_odds.pkl
  models/calibrator_top3_no_odds.pkl
  models/calibrator_top3_with_odds.pkl
  reports/GPT/outputs/charts/12_calibration_plot.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

from config import DATA_PROCESSED, MODELS_DIR, LOGS_DIR, REPORTS_CHARTS
from utils import get_logger, save_fig

logger = get_logger("calibration", LOGS_DIR)


def main():
    logger.info("=" * 60)
    logger.info("확률 보정 시작")
    logger.info("=" * 60)

    valid = load_split("valid")
    test  = load_split("test")

    if valid is None:
        logger.error("valid_fe.csv 없음")
        return

    cal_results = {}

    # ── No-Odds 분류기 보정 ────────────────────────────────────────────────
    for odds_mode in ["no_odds", "with_odds"]:
        for target_type in ["win", "top3"]:
            pred_path = DATA_PROCESSED / f"test_clf_{target_type}_{odds_mode}.csv"
            if not pred_path.exists():
                logger.warning(f"  없음: {pred_path.name} - 건너뜀")
                continue

            pred_test = pd.read_csv(pred_path, encoding="utf-8-sig", low_memory=False)
            prob_col  = f"{target_type}_prob"
            target_col = f"target_{target_type}"

            if prob_col not in pred_test.columns:
                logger.warning(f"  {prob_col} 컬럼 없음 - 건너뜀")
                continue

            # Valid set에서 보정 (분류기 예측 대신, Valid로 다시 추론 필요)
            # 여기서는 Test 예측값을 Valid 기반 calibrator로 보정
            model_name = f"classifier_{target_type}_{odds_mode}"
            model_path_cbm = MODELS_DIR / f"{model_name}.cbm"
            model_path_pkl = MODELS_DIR / f"{model_name}.pkl"

            valid_probs = None
            from utils import load_json
            from config import ARTIFACTS_DIR
            try:
                feat_info   = load_json(ARTIFACTS_DIR / f"{model_name}_features.json")
                features    = [f for f in feat_info["features"] if f in valid.columns]
                X_valid     = valid[features].fillna(-999)

                if model_path_cbm.exists():
                    from catboost import CatBoostClassifier
                    model = CatBoostClassifier()
                    model.load_model(str(model_path_cbm))
                    valid_probs = model.predict_proba(X_valid)[:, 1]
                elif model_path_pkl.exists():
                    with open(model_path_pkl, "rb") as f:
                        saved = pickle.load(f)
                    imp   = saved["imputer"]
                    model = saved["model"]
                    valid_probs = model.predict_proba(imp.transform(X_valid))[:, 1]
            except Exception as e:
                logger.warning(f"  Valid 재추론 실패: {e} - 보정 건너뜀")
                continue

            if valid_probs is None:
                continue

            y_valid_true = valid[target_col].values

            # ── Isotonic Regression 보정 ───────────────────────────────────
            try:
                from sklearn.calibration import CalibratedClassifierCV
                from sklearn.isotonic import IsotonicRegression

                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(valid_probs, y_valid_true)

                cal_name = f"calibrator_{target_type}_{odds_mode}"
                with open(MODELS_DIR / f"{cal_name}.pkl", "wb") as f:
                    pickle.dump(iso, f)

                # Test에 보정 적용
                raw_probs = pred_test[prob_col].values
                cal_probs = iso.predict(raw_probs)
                pred_test[f"{prob_col}_cal"] = cal_probs

                # 진단
                before_brier = _brier(y_valid_true, valid_probs)
                after_brier  = _brier(y_valid_true, iso.predict(valid_probs))
                logger.info(
                    f"  [{odds_mode}/{target_type}] Brier: "
                    f"보정전 {before_brier:.4f} → 보정후 {after_brier:.4f}"
                )
                cal_results[f"{target_type}_{odds_mode}"] = {
                    "brier_before": before_brier,
                    "brier_after":  after_brier,
                    "method": "isotonic",
                }

                # Race 내 확률 합 진단
                pred_test[f"{prob_col}_cal"] = cal_probs
                if target_type == "win":
                    race_sum = pred_test.groupby("race_id")[f"{prob_col}_cal"].sum()
                    logger.info(f"    race 내 win_prob 합: "
                                f"mean={race_sum.mean():.3f}, "
                                f"min={race_sum.min():.3f}, "
                                f"max={race_sum.max():.3f}")
                elif target_type == "top3":
                    race_sum = pred_test.groupby("race_id")[f"{prob_col}_cal"].sum()
                    logger.info(f"    race 내 top3_prob 합: "
                                f"mean={race_sum.mean():.3f} (기대값~3)")

                # 저장
                pred_test.to_csv(pred_path, index=False, encoding="utf-8-sig")
                logger.info(f"    보정값 저장: {pred_path.name}")

            except Exception as e:
                logger.warning(f"  Isotonic 보정 실패: {e}")

    # ── Calibration 차트 ───────────────────────────────────────────────────
    plot_calibration(pred_test if test is not None else None, valid, cal_results)

    logger.info("=" * 60)
    logger.info("보정 완료")
    return cal_results


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


def _brier(y_true, y_prob):
    from sklearn.metrics import brier_score_loss
    return round(brier_score_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7)), 4)


def plot_calibration(test_df, valid_df, cal_results):
    """보정 전/후 Brier Score 비교 차트"""
    if not cal_results:
        return

    labels  = list(cal_results.keys())
    before  = [cal_results[k]["brier_before"] for k in labels]
    after   = [cal_results[k]["brier_after"]  for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, before, width, label="보정 전", color="#E74C3C", alpha=0.85)
    ax.bar(x + width/2, after,  width, label="보정 후", color="#27AE60", alpha=0.85)
    ax.set_title("Calibration 전/후 Brier Score 비교", fontsize=13, fontweight="bold")
    ax.set_xlabel("모델")
    ax.set_ylabel("Brier Score (낮을수록 좋음)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - width/2, b + 0.001, f"{b:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width/2, a + 0.001, f"{a:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    save_fig(fig, REPORTS_CHARTS / "12_calibration_plot.png")
    logger.info("  12_calibration_plot.png 저장")


if __name__ == "__main__":
    main()
