from __future__ import annotations

import warnings

import pandas as pd
import plotly.express as px
import streamlit as st

from kra_race_prediction_stage04_streamlit_dashboard.src import utils
from kra_race_prediction_stage04_streamlit_dashboard.src.data_loader import first_existing_column, load_all_datasets
from kra_race_prediction_stage04_streamlit_dashboard.src.evaluation_view import (
    render_baseline_vs_v2,
    render_error_analysis,
    render_model_performance,
    render_walk_forward,
)
from kra_race_prediction_stage04_streamlit_dashboard.src.feature_view import render_feature_importance, render_feature_reasoning
from kra_race_prediction_stage04_streamlit_dashboard.src.prediction_service import load_model, predict_top3

warnings.filterwarnings("ignore")

st.set_page_config(page_title="KRA Top3 Prediction Prototype", layout="wide", page_icon="🐎")
utils.render_warning_disclaimer()
st.title("🏇 KRA 경주마 Top3 예측 대시보드")
st.caption("v2 모델 산출물과 Stage06 Walk-forward 검증 결과를 통합해 보여주는 분석용 프로토타입입니다.")

DATA = load_all_datasets()
MODEL = load_model() if utils.PATH_MODEL.exists() else None


def _status_message(status: dict) -> None:
    if not status:
        return
    level = status.get("level", "info")
    msg = status.get("message", "")
    if level == "success":
        st.success(msg)
    elif level == "error":
        st.error(msg)
    elif level == "warning":
        st.warning(msg)
    else:
        st.info(msg)


def _normalize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "pred_top3_prob" not in work.columns:
        for prob_col in ["top3_prob", "prob", "pred_prob", "prediction", "score"]:
            if prob_col in work.columns:
                work["pred_top3_prob"] = pd.to_numeric(work[prob_col], errors="coerce")
                break
    if "pred_rank_in_race" not in work.columns and "pred_rank" in work.columns:
        work["pred_rank_in_race"] = work["pred_rank"]
    if "pred_rank_in_race" not in work.columns and "pred_top3_prob" in work.columns:
        work["pred_rank_in_race"] = work.groupby("race_id")["pred_top3_prob"].rank(ascending=False, method="min") if "race_id" in work.columns else work["pred_top3_prob"].rank(ascending=False, method="min")
    if "pred_is_top3" not in work.columns and "pred_rank_in_race" in work.columns:
        rank = pd.to_numeric(work["pred_rank_in_race"], errors="coerce")
        work["pred_is_top3"] = (rank <= 3).fillna(False).astype(int)
    for col in ["target_is_top3", "is_top3"]:
        if col in work.columns:
            work["actual_top3"] = work[col]
            break
    return work


def _available_prediction_source() -> tuple[pd.DataFrame, str]:
    for key, label in [
        ("lgbm_pred", "v2 예측 결과"),
        ("modeling_ready", "v2 모델링 데이터"),
        ("historical_predictions", DATA.get("historical_prediction_source", "저장된 경주별 예측 결과")),
    ]:
        df = DATA.get(key, pd.DataFrame())
        if isinstance(df, pd.DataFrame) and not df.empty:
            return _normalize_prediction_columns(df), str(label)
    return pd.DataFrame(), ""


def _filter_race_ui(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    if df.empty:
        return df
    date_col = first_existing_column(df, ["schdRaceDt", "race_date", "date"])
    race_col = first_existing_column(df, ["race_id", "rc_id"])
    filtered = df
    if date_col:
        dates = sorted(filtered[date_col].dropna().astype(str).unique().tolist(), reverse=True)
        if dates:
            selected_date = st.selectbox("날짜 선택", dates, key=f"{key_prefix}_date")
            filtered = filtered[filtered[date_col].astype(str) == selected_date]
    if race_col:
        races = sorted(filtered[race_col].dropna().astype(str).unique().tolist())
        if races:
            selected_race = st.selectbox("race_id 선택", races, key=f"{key_prefix}_race")
            filtered = filtered[filtered[race_col].astype(str) == selected_race]
    return filtered.copy()


def _render_prediction_table(df: pd.DataFrame, mode: str) -> None:
    if df.empty:
        st.warning("표시할 예측 데이터가 없습니다.")
        return
    work = _normalize_prediction_columns(df)
    if mode == "historical" and MODEL is not None and "pred_top3_prob" not in work.columns:
        predicted = predict_top3(work, MODEL)
        if predicted is not None and not predicted.empty:
            work = predicted
    if "pred_top3_prob" in work.columns:
        prob = pd.to_numeric(work["pred_top3_prob"], errors="coerce")
        work["Top3 예상 확률(%)"] = (prob.where(prob > 1, prob * 100)).round(2)
    if "pred_rank_in_race" in work.columns:
        work["모델 예상 순위"] = pd.to_numeric(work["pred_rank_in_race"], errors="coerce")
    if "pred_is_top3" in work.columns:
        pred_flag = pd.to_numeric(work["pred_is_top3"], errors="coerce").fillna(0).astype(int)
        work["Top3 예상"] = pred_flag.apply(lambda x: "🏅 Top3" if x == 1 else "")
    if "actual_top3" in work.columns and mode == "historical":
        actual_flag = pd.to_numeric(work["actual_top3"], errors="coerce").fillna(0).astype(int)
        work["실제 Top3 여부"] = actual_flag.apply(lambda x: "✅ 실제 Top3" if x == 1 else "")
    elif mode == "historical":
        st.info("실제 결과 컬럼이 없어 정답 비교 없이 예측 결과만 표시합니다.")

    wanted = [
        "모델 예상 순위",
        "pthrGtno",
        "pthrHrno",
        "pthrHrnm",
        "hrmJckyNm",
        "hrmTrarNm",
        "pthrRatg",
        "pthrBurdWgt",
        "fe_horse_cum_avg_rk",
        "fe_jcky_cum_top3_rate",
        "fe_trar_cum_win_rate",
        "Top3 예상 확률(%)",
        "Top3 예상",
        "실제 Top3 여부",
    ]
    cols = [c for c in wanted if c in work.columns]
    if not cols:
        cols = work.columns.tolist()[:20]
    show = work.sort_values("모델 예상 순위") if "모델 예상 순위" in work.columns else work
    st.dataframe(utils.apply_friendly_columns(show[cols]), use_container_width=True, hide_index=True)

    if "pred_top3_prob" in work.columns:
        chart_df = work.copy()
        label_col = first_existing_column(chart_df, ["pthrHrnm", "pthrHrno", "horse_name"]) or chart_df.index.name or "index"
        if label_col == "index":
            chart_df = chart_df.reset_index()
        fig = px.bar(chart_df.sort_values("pred_top3_prob", ascending=True), x="pred_top3_prob", y=label_col, orientation="h", title="출전마별 Top3 예상 확률")
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)


def render_overview() -> None:
    st.header("프로젝트 개요")
    st.markdown(
        "본 시스템은 KRA 경주 데이터를 기반으로 경주마의 Top3 진입 가능성을 계산하고, "
        "모델 산출물·검증 결과·오류 분석을 한 화면에서 점검하기 위한 **분석용 프로토타입**입니다."
    )
    eval_df = DATA.get("eval_table", pd.DataFrame())
    if not eval_df.empty and eval_df["precision_at_3"].notna().any():
        best = eval_df.sort_values("precision_at_3", ascending=False).iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("최고 v2 Precision@3", best.get("precision_at_3_pct", "-"))
        c2.metric("최고 v2 Hit@3", best.get("hit_at_3_pct", "-"))
        c3.metric("모델", str(best.get("model_name", "-")))
    else:
        st.warning("v2 성능표를 읽지 못해 프로젝트 개요의 성능 수치를 표시하지 않습니다.")
    wf = DATA.get("walk_forward_monthly", pd.DataFrame())
    if not wf.empty:
        st.info(f"Stage06 Walk-forward 월별 지표 {len(wf):,}개 행이 로드되었습니다.")


def render_race_predictions() -> None:
    st.header("경주별 Top3 예측")
    mode = st.radio("예측 모드", ["과거 검증 모드", "미래 예측 모드"], horizontal=True)
    if mode == "미래 예측 모드":
        future = DATA.get("future_pred", pd.DataFrame())
        if future.empty:
            st.warning("미래 예측 파일이 없습니다. Stage05 파이프라인 실행 필요: `python run_inference.py`")
            return
        st.info("미래 경주 예측 모드입니다. 실제 결과가 없으므로 정답 비교는 표시하지 않습니다.")
        _render_prediction_table(_filter_race_ui(future, "future"), "future")
    else:
        pred_df, source = _available_prediction_source()
        if pred_df.empty:
            st.warning("과거 검증용 v2 예측/모델링 데이터가 없습니다. 데이터 상태 점검 탭에서 누락 파일을 확인하세요.")
            return
        st.caption(f"사용 데이터: {source}")
        if MODEL is None:
            st.info("모델 파일이 없어도 저장된 예측 CSV를 사용해 날짜별 예측 결과를 표시합니다.")
        _render_prediction_table(_filter_race_ui(pred_df, "hist"), "historical")


def render_feature_tab() -> None:
    st.header("예측 근거/피처 설명")
    pred_df, source = _available_prediction_source()
    if pred_df.empty:
        st.warning("예측 근거를 표시할 경주별 데이터가 없습니다.")
        return
    st.caption(f"사용 데이터: {source}")
    race_df = _filter_race_ui(pred_df, "feature")
    if race_df.empty:
        st.info("선택 가능한 말 데이터가 없습니다.")
        return
    horse_col = first_existing_column(race_df, ["pthrHrno", "pthrHrnm"])
    if not horse_col:
        render_feature_reasoning(race_df.head(1), DATA.get("feature_importances"))
        return
    options = race_df[horse_col].astype(str).tolist()
    selected = st.selectbox("분석 대상 말 선택", options)
    render_feature_reasoning(race_df[race_df[horse_col].astype(str) == selected], DATA.get("feature_importances"))


def render_file_status() -> None:
    st.header("데이터/파일 상태 점검")
    status_df = utils.check_file_status()
    st.dataframe(status_df, use_container_width=True, hide_index=True)
    missing = status_df[status_df["상태"].str.contains("없음")]
    if not missing.empty:
        st.warning("일부 파일이 없어 관련 탭 기능이 제한될 수 있습니다. 앱 전체는 중단하지 않습니다.")
    with st.expander("로드 상태 메시지"):
        for status in DATA.get("statuses", {}).values():
            _status_message(status)


tabs = st.tabs([
    "📂 1. 프로젝트 개요",
    "🎯 2. 경주별 Top3 예측",
    "💡 3. 예측 근거/피처 설명",
    "📊 4. v2 모델 성능 요약",
    "⚖️ 5. Baseline vs v2 비교",
    "🧪 6. Walk-forward 검증",
    "🔍 7. 변수 중요도",
    "🎯 8. 오류 분석",
    "🗂️ 9. 데이터/파일 상태 점검",
])

with tabs[0]:
    render_overview()
with tabs[1]:
    render_race_predictions()
with tabs[2]:
    render_feature_tab()
with tabs[3]:
    render_model_performance(DATA.get("eval_table", pd.DataFrame()))
with tabs[4]:
    render_baseline_vs_v2(DATA.get("eval_table", pd.DataFrame()), DATA.get("baseline_perf", pd.DataFrame()))
with tabs[5]:
    render_walk_forward(DATA.get("walk_forward_monthly", pd.DataFrame()))
with tabs[6]:
    source_label = "v2 feature_importance_v2.csv" if utils.PATH_FI.exists() else "fallback baseline lightgbm_feature_importance.csv"
    if not utils.PATH_FI.exists() and utils.PATH_BASE_FI.exists():
        st.warning("v2 변수 중요도 파일이 없어 baseline 변수 중요도 파일을 fallback으로 사용합니다.")
    render_feature_importance(DATA.get("feature_importances", pd.DataFrame()), source_label)
with tabs[7]:
    render_error_analysis(
        DATA.get("err_good", pd.DataFrame()),
        DATA.get("err_bad", pd.DataFrame()),
        DATA.get("err_distance", pd.DataFrame()),
        DATA.get("err_class", pd.DataFrame()),
        DATA.get("error_source", "baseline"),
    )
with tabs[8]:
    render_file_status()
