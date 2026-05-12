"""대시보드 데이터 로딩 및 스키마 정규화 유틸리티."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from . import utils

STANDARD_PERF_COLUMNS = [
    "model_name",
    "precision_at_3",
    "hit_at_3",
    "avg_correct",
    "top1_accuracy",
    "ndcg_at_3",
    "source",
    "note",
]


def _file_signature(path: Path | str) -> Tuple[str, bool, Optional[int], Optional[int]]:
    p = Path(path)
    if not p.exists():
        return str(p), False, None, None
    stat = p.stat()
    return str(p), True, stat.st_mtime_ns, stat.st_size


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        as_num = pd.to_numeric(df[col].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
        non_null_original = df[col].notna().sum()
        if non_null_original and as_num.notna().sum() / non_null_original >= 0.8:
            df[col] = as_num
    return df


@st.cache_data(show_spinner=False)
def _read_csv_cached(path_str: str, exists: bool, mtime_ns: Optional[int], size: Optional[int]) -> pd.DataFrame:
    if not exists:
        return pd.DataFrame()
    return pd.read_csv(path_str, encoding="utf-8-sig", low_memory=False)


def safe_read_csv(path: Path | str, required: bool = False, label: Optional[str] = None) -> tuple[pd.DataFrame, dict]:
    """CSV를 안전하게 읽고 (DataFrame, 상태 dict)를 반환한다."""
    p = Path(path)
    label = label or utils.rel_path(p)
    signature = _file_signature(p)
    if not signature[1]:
        return pd.DataFrame(), {"ok": False, "level": "error" if required else "warning", "message": f"{label} 파일 없음: {utils.rel_path(p)}"}
    try:
        df = _read_csv_cached(*signature)
        df = _coerce_numeric_columns(_clean_columns(df))
        return df, {"ok": True, "level": "success", "message": f"{label} 로드 완료 ({len(df):,}행)"}
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(p, encoding="cp949", low_memory=False)
            df = _coerce_numeric_columns(_clean_columns(df))
            return df, {"ok": True, "level": "success", "message": f"{label} 로드 완료(cp949, {len(df):,}행)"}
        except Exception as exc:  # noqa: BLE001
            return pd.DataFrame(), {"ok": False, "level": "error", "message": f"{label} 읽기 실패: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), {"ok": False, "level": "error", "message": f"{label} 읽기 실패: {exc}"}


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_probability_value(value):
    if pd.isna(value):
        return np.nan
    value = float(value)
    if abs(value) > 1:
        return value / 100
    return value


def percent_label(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:.2f}%"


def normalize_performance_table(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=STANDARD_PERF_COLUMNS + ["precision_at_3_pct", "hit_at_3_pct", "improvement_vs_baseline"])

    work = _clean_columns(df)
    lowered = {c.lower().strip().replace(" ", "_"): c for c in work.columns}

    def pick(*names):
        for name in names:
            key = name.lower().strip().replace(" ", "_")
            if key in lowered:
                return lowered[key]
        return None

    mapping = {
        "model_name": pick("model_name", "model name", "method"),
        "precision_at_3": pick("precision_at_3", "precision@3", "prec@3"),
        "hit_at_3": pick("hit_at_3", "hit@3"),
        "avg_correct": pick("avg_correct", "avg correct", "avg_correct_top3_count"),
        "top1_accuracy": pick("top1_accuracy", "top1_hit_rate", "top1 hit rate"),
        "ndcg_at_3": pick("ndcg_at_3", "ndcg@3"),
        "note": pick("note", "comment"),
    }

    out = pd.DataFrame(index=work.index)
    for std_col in STANDARD_PERF_COLUMNS:
        src_col = mapping.get(std_col)
        out[std_col] = work[src_col] if src_col else np.nan
    out["model_name"] = out["model_name"].fillna("Unknown")
    out["source"] = source or out["source"].fillna("")
    out["note"] = out["note"].fillna("")

    for col in ["precision_at_3", "hit_at_3", "top1_accuracy", "ndcg_at_3"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").map(normalize_probability_value)
    out["avg_correct"] = pd.to_numeric(out["avg_correct"], errors="coerce")

    baseline_mask = out["model_name"].astype(str).str.contains("baseline|stage 03|무작위|rule|규칙|평균순위", case=False, regex=True, na=False)
    baseline_precision = out.loc[baseline_mask, "precision_at_3"].dropna()
    baseline_value = baseline_precision.iloc[0] if not baseline_precision.empty else out["precision_at_3"].dropna().iloc[0] if out["precision_at_3"].notna().any() else np.nan
    out["improvement_vs_baseline"] = out["precision_at_3"] - baseline_value if not pd.isna(baseline_value) else np.nan
    out["precision_at_3_pct"] = out["precision_at_3"].map(percent_label)
    out["hit_at_3_pct"] = out["hit_at_3"].map(percent_label)
    out["top1_accuracy_pct"] = out["top1_accuracy"].map(percent_label)
    out["ndcg_at_3_pct"] = out["ndcg_at_3"].map(percent_label)
    out["improvement_pctp"] = out["improvement_vs_baseline"].map(lambda x: "-" if pd.isna(x) else f"{x * 100:+.2f}%p")
    return out.reset_index(drop=True)


def normalize_walk_forward_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_performance_table(df.assign(model_name=df.get("test_month", "Walk-forward")), source="Stage06 Walk-forward")
    if df is not None and not df.empty:
        for col in ["test_month", "n_train_rows", "n_test_rows", "rank_mae"]:
            if col in df.columns:
                out[col] = df[col].values
    return out


def load_all_datasets() -> dict:
    datasets: dict = {"statuses": {}}
    specs = {
        "modeling_ready": (utils.PATH_MOD_READY, False, "v2 모델링 데이터"),
        "lgbm_pred": (utils.PATH_LGBM_PRED, False, "v2 예측 결과"),
        "future_pred": (utils.PATH_NEXT_PRED, False, "Stage05 미래 예측 결과"),
        "eval_table_raw": (utils.PATH_CMP_TBL, True, "v2 성능 비교표"),
        "baseline_perf_raw": (utils.PATH_BASE_PERF, False, "baseline 성능표"),
        "feature_importances": (utils.PATH_FI if utils.PATH_FI.exists() else utils.PATH_BASE_FI, False, "변수 중요도"),
        "walk_forward_monthly_raw": (utils.PATH_WF_MONTHLY, True, "Stage06 월별 지표"),
        "walk_forward_predictions": (utils.PATH_WF_PREDS, False, "Stage06 전체 예측"),
    }
    for key, (path, required, label) in specs.items():
        datasets[key], datasets["statuses"][key] = safe_read_csv(path, required=required, label=label)

    datasets["eval_table"] = normalize_performance_table(datasets["eval_table_raw"], source="Stage03 v2")
    datasets["baseline_perf"] = normalize_performance_table(datasets["baseline_perf_raw"], source="Stage03 baseline")
    datasets["walk_forward_monthly"] = normalize_walk_forward_metrics(datasets["walk_forward_monthly_raw"])

    err_paths = utils.existing_error_paths()
    datasets["error_source"] = "v2" if err_paths["good"] == utils.PATH_V2_ERR_GOOD else "baseline"
    for key, path in err_paths.items():
        datasets[f"err_{key}"], datasets["statuses"][f"err_{key}"] = safe_read_csv(path, required=False, label=f"오류 분석({key})")
    return datasets
