"""
Streamlit 대시보드 공통 유틸리티 및 프로젝트 경로 설정.

모든 산출물 경로는 프로젝트 루트를 기준으로 한 곳에서 관리한다. 파일이
누락되어도 앱 전체가 중단되지 않도록 상태 점검용 메타데이터를 제공한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent
STAGE04_DIR = SRC_DIR.parent
PROJECT_ROOT = STAGE04_DIR.parent

STAGE03_V2_DIR = PROJECT_ROOT / "kra_race_prediction_stage03_model_upgrade_v2"
STAGE03_BASELINE_DIR = PROJECT_ROOT / "kra_race_prediction_stage03_top3_modeling"
STAGE05_DIR = PROJECT_ROOT / "kra_race_prediction_stage05_inference_pipeline"
STAGE06_DIR = PROJECT_ROOT / "kra_race_prediction_stage06_walk_forward_backtest"

PATH_MODEL = STAGE03_V2_DIR / "models" / "lgbm_top3_feature_v2.pkl"
PATH_MOD_READY = STAGE03_V2_DIR / "data" / "modeling" / "modeling_data_v2_with_preds.csv"
PATH_LGBM_PRED = STAGE03_V2_DIR / "data" / "predictions" / "lgbm_top3_feature_v2_predictions.csv"
PATH_CMP_TBL = STAGE03_V2_DIR / "outputs" / "tables" / "model_v2_comparison_table.csv"
PATH_FI = STAGE03_V2_DIR / "outputs" / "tables" / "feature_importance_v2.csv"
PATH_V2_ERR_GOOD = STAGE03_V2_DIR / "outputs" / "tables" / "good_prediction_races_v2.csv"
PATH_V2_ERR_BAD = STAGE03_V2_DIR / "outputs" / "tables" / "bad_prediction_races_v2.csv"
PATH_V2_ERR_DIST = STAGE03_V2_DIR / "outputs" / "tables" / "error_analysis_by_distance_v2.csv"
PATH_V2_ERR_CLS = STAGE03_V2_DIR / "outputs" / "tables" / "error_analysis_by_class_v2.csv"

PATH_NEXT_PRED = STAGE05_DIR / "data" / "output" / "next_race_predictions.csv"

PATH_BASE_PRED = STAGE03_BASELINE_DIR / "data" / "predictions" / "baseline_rule_predictions.csv"
PATH_BASE_PERF = STAGE03_BASELINE_DIR / "outputs" / "metrics" / "model_performance_summary.csv"
PATH_BASE_FI = STAGE03_BASELINE_DIR / "outputs" / "tables" / "lightgbm_feature_importance.csv"
PATH_ERR_GOOD = STAGE03_BASELINE_DIR / "outputs" / "tables" / "good_prediction_races.csv"
PATH_ERR_BAD = STAGE03_BASELINE_DIR / "outputs" / "tables" / "bad_prediction_races.csv"
PATH_ERR_DIST = STAGE03_BASELINE_DIR / "outputs" / "tables" / "error_analysis_by_distance.csv"
PATH_ERR_CLS = STAGE03_BASELINE_DIR / "outputs" / "tables" / "error_analysis_by_class.csv"

PATH_WF_MONTHLY = STAGE06_DIR / "outputs" / "metrics" / "walk_forward_monthly_metrics.csv"
PATH_WF_PREDS = STAGE06_DIR / "data" / "predictions" / "walk_forward_predictions_all.csv"
PATH_WF_REPORT = STAGE06_DIR / "reports" / "walk_forward_report.md"
PATH_WF_FIGURES = STAGE06_DIR / "outputs" / "figures"


@dataclass(frozen=True)
class FileSpec:
    key: str
    label: str
    path: Path
    required: bool = False
    feature: str = ""


CORE_FILE_SPECS = [
    FileSpec("v2_model", "v2 모델 파일", PATH_MODEL, False, "v2 예측 모델 로딩"),
    FileSpec("v2_modeling_data", "v2 모델링 데이터", PATH_MOD_READY, False, "과거 검증 모드"),
    FileSpec("v2_predictions", "v2 예측 결과", PATH_LGBM_PRED, False, "과거 예측 결과 표시"),
    FileSpec("v2_comparison", "v2 성능 비교표", PATH_CMP_TBL, True, "v2 모델 성능 요약"),
    FileSpec("v2_feature_importance", "v2 변수 중요도", PATH_FI, True, "변수 중요도 탭"),
    FileSpec("stage05_next_predictions", "Stage05 미래 예측 결과", PATH_NEXT_PRED, False, "미래 예측 모드"),
    FileSpec("stage06_monthly_metrics", "Stage06 월별 Walk-forward 지표", PATH_WF_MONTHLY, True, "Walk-forward 검증"),
    FileSpec("stage06_all_predictions", "Stage06 전체 Walk-forward 예측 결과", PATH_WF_PREDS, False, "Walk-forward 상세 분석"),
    FileSpec("stage06_report", "Stage06 보고서", PATH_WF_REPORT, False, "Walk-forward 보고서"),
    FileSpec("baseline_rule_predictions", "baseline rule prediction 파일", PATH_BASE_PRED, False, "Baseline vs v2 비교"),
    FileSpec("baseline_error_good", "baseline good races 파일", PATH_ERR_GOOD, False, "오류 분석"),
    FileSpec("baseline_error_bad", "baseline bad races 파일", PATH_ERR_BAD, False, "오류 분석"),
    FileSpec("baseline_error_distance", "baseline error analysis by distance 파일", PATH_ERR_DIST, False, "오류 분석"),
    FileSpec("baseline_error_class", "baseline error analysis by class 파일", PATH_ERR_CLS, False, "오류 분석"),
]

TERMINOLOGY_MAP = {
    "pred_top3_prob": "Top3 예상 확률",
    "top3_prob": "Top3 예상 확률",
    "pred_rank_in_race": "모델 예상 순위",
    "pred_rank": "모델 예상 순위",
    "pred_is_top3": "모델 Top3 선택",
    "target_is_top3": "실제 Top3 여부",
    "is_top3": "실제 Top3 여부",
    "pthrGtno": "게이트번호",
    "pthrHrno": "출전번호",
    "pthrHrnm": "마명",
    "hrmJckyNm": "기수",
    "hrmTrarNm": "조교사",
    "pthrRatg": "레이팅",
    "pthrBurdWgt": "부담중량",
    "fe_horse_cum_avg_rk": "말 과거 평균순위",
    "fe_jcky_cum_top3_rate": "기수 누적 Top3율",
    "fe_jcky_cum_win_rate": "기수 누적 승률",
    "fe_trar_cum_win_rate": "조교사 누적 승률",
    "horse_avg_rank_rank_in_race": "경주 내 과거 성적 순위",
    "jockey_top3_rate_rank_in_race": "경주 내 기수 Top3율 순위",
    "jockey_winrate_rank_in_race": "경주 내 기수 승률 순위",
    "trainer_winrate_rank_in_race": "경주 내 조교사 승률 순위",
    "rating_zscore_in_race": "경주 내 레이팅 우위",
    "weight_zscore_in_race": "경주 내 부담중량 편차",
}

FEATURE_DESCRIPTION_MAP = {
    "fe_horse_cum_adj_rank_score_std": "말의 누적 보정 순위 점수 변동성",
    "fe_trar_cum_win_rate": "조교사의 누적 승률",
    "fe_horse_weight": "말 체중",
    "fe_horse_days_since_last_race": "직전 경주 후 경과일",
    "fe_jcky_cum_win_rate": "기수의 누적 승률",
    "horse_avg_rank_rank_in_race": "같은 경주 내 말의 과거 평균순위 상대 순위",
    "fe_horse_cum_avg_rk": "말의 과거 평균 순위",
    "jockey_top3_rate_rank_in_race": "같은 경주 내 기수 Top3율 상대 순위",
    "weight_zscore_in_race": "부담중량이 경주 평균에서 벗어난 정도",
    "rating_zscore_in_race": "레이팅이 경주 평균에서 벗어난 정도",
    "fe_horse_race_count": "말의 누적 출전 횟수",
    "pthrRatg": "공식 레이팅",
    "pthrBurdWgt": "부담중량",
}


def rel_path(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def translate_term(internal_term: str) -> str:
    return TERMINOLOGY_MAP.get(internal_term, internal_term)


def apply_friendly_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=TERMINOLOGY_MAP)


def check_file_status(specs: Optional[Iterable[FileSpec]] = None) -> pd.DataFrame:
    rows = []
    for spec in specs or CORE_FILE_SPECS:
        exists = spec.path.exists()
        rows.append(
            {
                "key": spec.key,
                "파일": spec.label,
                "상태": "✅ 있음" if exists else "⚠️ 없음",
                "필수 여부": "권장/핵심" if spec.required else "선택/부가",
                "제한되는 기능": spec.feature,
                "상대 경로": rel_path(spec.path),
                "크기(KB)": round(spec.path.stat().st_size / 1024, 1) if exists and spec.path.is_file() else None,
            }
        )
    return pd.DataFrame(rows)


def existing_error_paths() -> Dict[str, Path]:
    """v2 오류 분석 파일이 있으면 우선 사용하고, 없으면 baseline으로 대체."""
    v2_paths = {
        "good": PATH_V2_ERR_GOOD,
        "bad": PATH_V2_ERR_BAD,
        "distance": PATH_V2_ERR_DIST,
        "class": PATH_V2_ERR_CLS,
    }
    if any(path.exists() for path in v2_paths.values()):
        return v2_paths
    return {"good": PATH_ERR_GOOD, "bad": PATH_ERR_BAD, "distance": PATH_ERR_DIST, "class": PATH_ERR_CLS}


def render_warning_disclaimer() -> None:
    st.warning(
        "⚠️ **주의**: 본 대시보드는 경주 전 데이터 기반 Top3 진입 가능성을 분석하기 위한 "
        "프로토타입입니다. 실제 경주 결과를 보장하지 않으며, 도박 또는 베팅 권유 목적이 아닙니다."
    )
