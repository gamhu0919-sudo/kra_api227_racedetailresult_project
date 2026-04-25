# -*- coding: utf-8 -*-
"""
evaluate.py - 통합 성능 평가 및 앙상블 최종 점수 설계
실행: python src/evaluate.py

산출물:
  reports/GPT/outputs/tables/13_model_comparison.csv
  reports/GPT/outputs/tables/14_monthly_performance.csv
  reports/GPT/outputs/tables/15_sample_predictions.csv
  reports/GPT/outputs/charts/10_feature_importance_no_odds.png
  reports/GPT/outputs/charts/11_feature_importance_with_odds.png
  reports/GPT/outputs/charts/13_model_comparison.png
  data/processed/final_predictions.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from config import DATA_PROCESSED, REPORTS_TABLES, REPORTS_CHARTS, MODELS_DIR, LOGS_DIR
from utils import (get_logger, save_fig, ndcg_at_k, spearman_by_race,
                   rank_within_race, normalize_within_race)

logger = get_logger("evaluate", LOGS_DIR)


def main():
    logger.info("=" * 60)
    logger.info("통합 평가 시작")
    logger.info("=" * 60)

    # 베이스라인 결과 로드
    baseline_df = safe_load_csv(REPORTS_TABLES / "10_baseline_results.csv")
    ranker_df   = safe_load_csv(REPORTS_TABLES / "11_ranker_results.csv")
    clf_df      = safe_load_csv(REPORTS_TABLES / "12_classifier_results.csv")

    # 테스트 예측결과 로드
    test_base   = load_split("test")
    ranker_no   = safe_load_csv(DATA_PROCESSED / "test_ranker_no_odds.csv")
    ranker_with = safe_load_csv(DATA_PROCESSED / "test_ranker_with_odds.csv")
    clf_win_no  = safe_load_csv(DATA_PROCESSED / "test_clf_win_no_odds.csv")
    clf_top3_no = safe_load_csv(DATA_PROCESSED / "test_clf_top3_no_odds.csv")
    clf_win_w   = safe_load_csv(DATA_PROCESSED / "test_clf_win_with_odds.csv")
    clf_top3_w  = safe_load_csv(DATA_PROCESSED / "test_clf_top3_with_odds.csv")

    all_results = []

    # ── 베이스라인 결과 통합 ──────────────────────────────────────────────
    if baseline_df is not None:
        for _, row in baseline_df.iterrows():
            all_results.append({
                "model": row.get("model", "baseline"),
                "leakage_safe": "Y",
                "uses_odds": row.get("uses_odds", "?"),
                "winner_hit_rate": row.get("winner_hit_rate"),
                "top3_hit_rate":   row.get("top3_hit_rate"),
                "ndcg_3":          row.get("ndcg_3"),
                "ndcg_5":          row.get("ndcg_5"),
                "spearman":        row.get("spearman"),
                "rank_mae":        row.get("rank_mae"),
                "brier_win":       row.get("brier_win"),
                "logloss_win":     row.get("logloss_win"),
            })

    # ── 앙상블 최종 점수 설계 ─────────────────────────────────────────────
    logger.info("\n[앙상블 최종 점수 설계]")

    # No-Odds 앙상블
    final_no = build_ensemble(
        test_base, ranker_no, clf_win_no, clf_top3_no,
        mode="no_odds", w_rank=0.5, w_win=0.3, w_top3=0.2
    )
    if final_no is not None:
        res = full_evaluate(final_no, "final_score", "pred_rank", "win_prob", "top3_prob")
        all_results.append({"model": "Ensemble_NoOdds", "leakage_safe": "Y",
                             "uses_odds": "N", **res})
        final_no.to_csv(DATA_PROCESSED / "final_predictions_no_odds.csv",
                        index=False, encoding="utf-8-sig")
        logger.info(f"  No-Odds 앙상블: {res}")

    # With-Odds 앙상블
    final_with = build_ensemble(
        test_base, ranker_with, clf_win_w, clf_top3_w,
        mode="with_odds", w_rank=0.4, w_win=0.35, w_top3=0.25
    )
    if final_with is not None:
        res = full_evaluate(final_with, "final_score", "pred_rank", "win_prob", "top3_prob")
        all_results.append({"model": "Ensemble_WithOdds", "leakage_safe": "Y",
                             "uses_odds": "Y", **res})
        final_with.to_csv(DATA_PROCESSED / "final_predictions_with_odds.csv",
                          index=False, encoding="utf-8-sig")
        logger.info(f"  With-Odds 앙상블: {res}")

    # 최종 예측 합본
    best_final = final_no if final_no is not None else final_with
    if best_final is not None:
        best_final.to_csv(DATA_PROCESSED / "final_predictions.csv",
                          index=False, encoding="utf-8-sig")

    # ── 모델 비교표 ───────────────────────────────────────────────────────
    comp_df = pd.DataFrame(all_results)
    comp_df.to_csv(REPORTS_TABLES / "13_model_comparison.csv",
                   index=False, encoding="utf-8-sig")
    logger.info(f"\n모델 비교표:\n{comp_df.to_string(index=False)}")

    # ── 월별 성능 ─────────────────────────────────────────────────────────
    if best_final is not None and "rcDate" in best_final.columns:
        monthly_perf = compute_monthly_performance(best_final)
        monthly_perf.to_csv(REPORTS_TABLES / "14_monthly_performance.csv",
                             index=False, encoding="utf-8-sig")
        logger.info("월별 성능 저장: 14_monthly_performance.csv")

    # ── 샘플 예측 테이블 ──────────────────────────────────────────────────
    if best_final is not None:
        sample = make_sample_table(best_final)
        sample.to_csv(REPORTS_TABLES / "15_sample_predictions.csv",
                      index=False, encoding="utf-8-sig")
        logger.info("샘플 예측 저장: 15_sample_predictions.csv")

    # ── 차트 생성 ─────────────────────────────────────────────────────────
    plot_feature_importance("no_odds")
    plot_feature_importance("with_odds")
    plot_model_comparison(comp_df)

    logger.info("=" * 60)
    logger.info("평가 완료")
    return comp_df


# ── 앙상블 ───────────────────────────────────────────────────────────────
def build_ensemble(test_base, ranker_df, clf_win_df, clf_top3_df,
                   mode, w_rank, w_win, w_top3):
    """
    final_score = w_rank * norm(rank_score) + w_win * win_prob + w_top3 * top3_prob
    """
    if test_base is None:
        return None

    result = test_base.copy()
    prob_col_raw = "win_prob"
    top3_col_raw = "top3_prob"

    # rank_score 병합
    if ranker_df is not None and "rank_score" in ranker_df.columns:
        rank_merge = ranker_df[["race_id", "chulNo", "rank_score"]].copy()
        result = result.merge(rank_merge, on=["race_id", "chulNo"], how="left")
        result = normalize_within_race(result, "rank_score")
        rank_norm = result["rank_score_norm"].fillna(0.5)
    else:
        rank_norm = pd.Series(0.5, index=result.index)
        logger.warning(f"  [{mode}] ranker 결과 없음 - rank_score 0.5 대체")

    # win_prob 병합
    if clf_win_df is not None:
        prob_col = "win_prob_cal" if "win_prob_cal" in clf_win_df.columns else "win_prob"
        if prob_col in clf_win_df.columns:
            win_merge = clf_win_df[["race_id", "chulNo", prob_col]].copy()
            result = result.merge(win_merge.rename(columns={prob_col: "win_prob"}),
                                  on=["race_id", "chulNo"], how="left")
            win_p = result["win_prob"].fillna(0.1)
        else:
            win_p = pd.Series(0.1, index=result.index)
    else:
        win_p = pd.Series(0.1, index=result.index)

    # top3_prob 병합
    if clf_top3_df is not None:
        prob_col = "top3_prob_cal" if "top3_prob_cal" in clf_top3_df.columns else "top3_prob"
        if prob_col in clf_top3_df.columns:
            top3_merge = clf_top3_df[["race_id", "chulNo", prob_col]].copy()
            result = result.merge(top3_merge.rename(columns={prob_col: "top3_prob"}),
                                  on=["race_id", "chulNo"], how="left")
            top3_p = result["top3_prob"].fillna(0.2)
        else:
            top3_p = pd.Series(0.2, index=result.index)
    else:
        top3_p = pd.Series(0.2, index=result.index)

    result["win_prob"]  = win_p.values
    result["top3_prob"] = top3_p.values

    # final_score
    result["final_score"] = (
        w_rank * rank_norm.values
        + w_win  * result["win_prob"].values
        + w_top3 * result["top3_prob"].values
    )

    # pred_rank
    result["pred_rank"] = rank_within_race(result, "final_score", ascending=False)

    return result


# ── 통합 평가 ─────────────────────────────────────────────────────────────
def full_evaluate(df, score_col, pred_rank_col, win_prob_col, top3_prob_col):
    from sklearn.metrics import brier_score_loss, log_loss
    df2 = df.dropna(subset=["ord"]).copy()

    top1     = df2[df2[pred_rank_col] == 1]
    win_hr   = (top1["ord"] == 1).mean() if len(top1) else 0.0
    top3_p_d = df2[df2[pred_rank_col] <= 3]
    top3_hr  = (top3_p_d["ord"] <= 3).mean() if len(top3_p_d) else 0.0

    ndcg3 = ndcg_at_k(df2, score_col, "ord", k=3)
    ndcg5 = ndcg_at_k(df2, score_col, "ord", k=5)
    spear = spearman_by_race(df2, pred_rank_col, "ord")
    r_mae = abs(df2[pred_rank_col] - df2["ord"]).mean()

    result = {
        "winner_hit_rate": round(win_hr, 4),
        "top3_hit_rate":   round(top3_hr, 4),
        "ndcg_3":          round(ndcg3, 4),
        "ndcg_5":          round(ndcg5, 4),
        "spearman":        round(spear, 4),
        "rank_mae":        round(r_mae, 4),
    }

    if win_prob_col in df2.columns:
        y_true = df2["target_win"].values
        y_prob = np.clip(df2[win_prob_col].values, 1e-7, 1 - 1e-7)
        result["brier_win"]   = round(brier_score_loss(y_true, y_prob), 4)
        result["logloss_win"] = round(log_loss(y_true, y_prob), 4)

    return result


# ── 월별 성능 ─────────────────────────────────────────────────────────────
def compute_monthly_performance(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.dropna(subset=["ord"]).copy()
    df2["year_month"] = (df2["rcDate"] // 100).astype(str)
    rows = []
    for ym, grp in df2.groupby("year_month"):
        top1   = grp[grp["pred_rank"] == 1]
        top3_p = grp[grp["pred_rank"] <= 3]
        win_hr = (top1["ord"] == 1).mean() if len(top1) else np.nan
        t3_hr  = (top3_p["ord"] <= 3).mean() if len(top3_p) else np.nan
        rows.append({
            "year_month":      ym,
            "n_races":         grp["race_id"].nunique(),
            "n_entries":       len(grp),
            "winner_hit_rate": round(win_hr, 4) if not np.isnan(win_hr) else None,
            "top3_hit_rate":   round(t3_hr, 4) if not np.isnan(t3_hr) else None,
        })
    return pd.DataFrame(rows)


# ── 샘플 예측 테이블 ──────────────────────────────────────────────────────
def make_sample_table(df: pd.DataFrame, n_races: int = 5) -> pd.DataFrame:
    """최근 n개 경주의 예측 결과 샘플"""
    recent_races = df.sort_values("rcDate", ascending=False)["race_id"].unique()[:n_races]
    sample = df[df["race_id"].isin(recent_races)].copy()
    cols = ["race_id", "rcDate", "rcNo", "chulNo",
            "hrName", "jkName", "trName",
            "pred_rank", "final_score", "win_prob", "top3_prob", "ord"]
    return sample[[c for c in cols if c in sample.columns]].sort_values(
        ["race_id", "pred_rank"]
    )


# ── 피처 중요도 차트 ──────────────────────────────────────────────────────
def plot_feature_importance(mode: str):
    fi_path = MODELS_DIR / f"ranker_{mode}_feature_importance.csv"
    if not fi_path.exists():
        fi_path = MODELS_DIR / f"classifier_win_{mode}_feature_importance.csv"
    if not fi_path.exists():
        logger.warning(f"  {mode} 피처 중요도 파일 없음")
        return

    fi = pd.read_csv(fi_path, encoding="utf-8-sig")
    fi = fi.nlargest(20, "importance")

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(fi)))[::-1]
    ax.barh(fi["feature"][::-1], fi["importance"][::-1],
            color=colors, edgecolor="white")
    title = f"피처 중요도 상위 20 ({'No-Odds' if mode == 'no_odds' else 'With-Odds'})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("중요도(Gain)")
    plt.tight_layout()
    save_fig(fig, REPORTS_CHARTS / f"1{'0' if mode=='no_odds' else '1'}_feature_importance_{mode}.png")
    logger.info(f"  Feature Importance 차트 저장: {mode}")


# ── 모델 비교 차트 ────────────────────────────────────────────────────────
def plot_model_comparison(comp_df: pd.DataFrame):
    if comp_df.empty:
        return

    df2 = comp_df.dropna(subset=["winner_hit_rate"]).copy()
    if df2.empty:
        return

    x = np.arange(len(df2))
    metrics = ["winner_hit_rate", "top3_hit_rate"]
    colors  = ["#2E86AB", "#A23B72"]

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.35
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = pd.to_numeric(df2[metric], errors="coerce").fillna(0)
        ax.bar(x + i * width, vals, width, label=metric, color=color, alpha=0.85)
        for xi, v in zip(x + i * width, vals):
            ax.text(xi, float(v) + 0.005, f"{float(v):.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_title("모델별 Hit Rate 비교", fontsize=13, fontweight="bold")
    ax.set_xlabel("모델")
    ax.set_ylabel("Hit Rate")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(df2["model"].values, rotation=25, ha="right", fontsize=9)
    ax.legend()
    plt.tight_layout()
    save_fig(fig, REPORTS_CHARTS / "13_model_comparison.png")
    logger.info("  모델 비교 차트 저장")


def safe_load_csv(path: Path):
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception:
        return None


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


if __name__ == "__main__":
    main()
