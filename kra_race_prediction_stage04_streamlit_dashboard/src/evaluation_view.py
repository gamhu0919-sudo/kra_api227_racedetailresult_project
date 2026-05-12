"""모델 성능, Walk-forward 검증, 오류 분석 View."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from . import utils
from .data_loader import percent_label


def _display_perf_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    cols = [
        "model_name",
        "precision_at_3_pct",
        "hit_at_3_pct",
        "avg_correct",
        "top1_accuracy_pct",
        "ndcg_at_3_pct",
        "improvement_pctp",
        "source",
        "note",
    ]
    display = display[[c for c in cols if c in display.columns]]
    return display.rename(
        columns={
            "model_name": "모델명",
            "precision_at_3_pct": "Precision@3",
            "hit_at_3_pct": "Hit@3",
            "avg_correct": "Avg Correct",
            "top1_accuracy_pct": "Top1 Accuracy",
            "ndcg_at_3_pct": "NDCG@3",
            "improvement_pctp": "baseline 대비 개선폭",
            "source": "출처",
            "note": "비고",
        }
    )


def render_model_performance(eval_df: pd.DataFrame) -> None:
    st.subheader("📊 v2 모델 성능 요약")
    model_exists = utils.PATH_MODEL.exists()
    st.info(f"현재 대시보드는 v2 모델 파일을 기준으로 구성됩니다: `{utils.rel_path(utils.PATH_MODEL)}`")
    st.write("모델 파일 상태:", "✅ 존재" if model_exists else "⚠️ 파일 없음(대시보드는 산출 CSV 기반으로 계속 표시)")

    if eval_df is None or eval_df.empty or eval_df["precision_at_3"].isna().all():
        st.warning("성능표 파일 없음/스키마 불일치로 v2 모델 성능 요약을 표시할 수 없습니다.")
        return

    best = eval_df.sort_values("precision_at_3", ascending=False).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("최고 Precision@3", percent_label(best["precision_at_3"]), best.get("improvement_pctp", ""))
    c2.metric("최고 모델", str(best["model_name"]))
    c3.metric("Hit@3", percent_label(best.get("hit_at_3")))
    c4.metric("Avg Correct", f"{best.get('avg_correct', float('nan')):.3f}" if pd.notna(best.get("avg_correct")) else "-")

    st.dataframe(_display_perf_table(eval_df), use_container_width=True, hide_index=True)

    fig1 = px.bar(eval_df, x="model_name", y="precision_at_3", text="precision_at_3_pct", title="Precision@3 비교")
    fig1.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(eval_df, x="model_name", y="hit_at_3", text="hit_at_3_pct", title="Hit@3 비교")
    fig2.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.bar(eval_df, x="model_name", y="avg_correct", title="Avg Correct 비교")
    st.plotly_chart(fig3, use_container_width=True)


def render_baseline_vs_v2(eval_df: pd.DataFrame, baseline_df: pd.DataFrame) -> None:
    st.subheader("⚖️ Baseline vs v2 비교")
    frames = [df for df in [baseline_df, eval_df] if df is not None and not df.empty]
    if not frames:
        st.warning("비교 가능한 성능표가 없습니다.")
        return
    combined = pd.concat(frames, ignore_index=True)
    st.dataframe(_display_perf_table(combined), use_container_width=True, hide_index=True)
    fig = px.bar(combined, x="model_name", y="precision_at_3", color="source", text="precision_at_3_pct", title="Baseline/Rule/v2 Precision@3")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("서로 다른 평가 기간·분할 방식의 수치는 해석에 주의가 필요합니다.")


def render_walk_forward(monthly_df: pd.DataFrame) -> None:
    st.subheader("🧪 Walk-forward 검증")
    st.markdown(
        "- 테스트 월 이전 데이터만 학습에 사용\n"
        "- 월별 누적 학습으로 검증\n"
        "- 미래 데이터 누수 방지를 목적으로 한 운영형 검증 방식"
    )
    st.warning("Stage03 단순 고정분할 성능과 Stage06 Walk-forward 성능은 평가 방식이 다르므로 직접 수치를 단순 비교하면 안 됩니다.")

    if monthly_df is None or monthly_df.empty:
        st.warning("Stage06 월별 Walk-forward 지표 파일이 없어 차트와 요약을 표시할 수 없습니다.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("평균 Precision@3", percent_label(monthly_df["precision_at_3"].mean()))
    c2.metric("평균 Hit@3", percent_label(monthly_df["hit_at_3"].mean()))
    top1_col = "top1_accuracy" if "top1_accuracy" in monthly_df.columns else None
    c3.metric("평균 Top1 Hit Rate", percent_label(monthly_df[top1_col].mean()) if top1_col else "-")
    ndcg_or_mae = "ndcg_at_3" if "ndcg_at_3" in monthly_df.columns else "rank_mae" if "rank_mae" in monthly_df.columns else None
    c4.metric("평균 NDCG@3/Rank MAE", f"{monthly_df[ndcg_or_mae].mean():.3f}" if ndcg_or_mae else "-")

    x_col = "test_month" if "test_month" in monthly_df.columns else "model_name"
    fig_p = px.line(monthly_df, x=x_col, y="precision_at_3", markers=True, title="월별 Precision@3")
    fig_p.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_p, use_container_width=True)

    fig_h = px.line(monthly_df, x=x_col, y="hit_at_3", markers=True, title="월별 Hit@3")
    fig_h.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_h, use_container_width=True)

    if "n_train_rows" in monthly_df.columns:
        fig_s = px.scatter(monthly_df, x="n_train_rows", y="precision_at_3", color=x_col, size="n_test_rows" if "n_test_rows" in monthly_df.columns else None, title="학습 데이터 크기 대비 Precision@3")
        fig_s.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_s, use_container_width=True)

    cols = [x_col, "precision_at_3_pct", "hit_at_3_pct", "top1_accuracy_pct", "ndcg_at_3_pct", "n_train_rows", "n_test_rows"]
    show_cols = [c for c in cols if c in monthly_df.columns]
    st.markdown("#### 성능 상위 5개월")
    st.dataframe(monthly_df.sort_values("precision_at_3", ascending=False)[show_cols].head(5), use_container_width=True, hide_index=True)
    st.markdown("#### 성능 하위 5개월")
    st.dataframe(monthly_df.sort_values("precision_at_3", ascending=True)[show_cols].head(5), use_container_width=True, hide_index=True)

    if utils.PATH_WF_REPORT.exists():
        with st.expander("Walk-forward 보고서 보기"):
            st.markdown(utils.PATH_WF_REPORT.read_text(encoding="utf-8"))
    if utils.PATH_WF_FIGURES.exists():
        pngs = sorted(Path(utils.PATH_WF_FIGURES).glob("*.png"))
        if pngs:
            st.markdown("#### Stage06 저장 이미지")
            for png in pngs[:4]:
                st.image(str(png), caption=utils.rel_path(png), use_container_width=True)


def render_error_analysis(good_df: pd.DataFrame, bad_df: pd.DataFrame, dist_df: pd.DataFrame, cls_df: pd.DataFrame, source: str = "baseline") -> None:
    st.subheader("🎯 오류 분석")
    st.caption(f"사용 오류 분석 소스: {source}")
    if all(df is None or df.empty for df in [good_df, bad_df, dist_df, cls_df]):
        st.info("오류 분석 파일 없음")
        return

    c1, c2 = st.columns(2)
    c1.metric("Good races 수", 0 if good_df is None else len(good_df))
    c2.metric("Bad races 수", 0 if bad_df is None else len(bad_df))

    c3, c4 = st.columns(2)
    with c3:
        if dist_df is not None and not dist_df.empty:
            y = "hit_ratio" if "hit_ratio" in dist_df.columns else "avg_correct" if "avg_correct" in dist_df.columns else None
            if y:
                fig = px.bar(dist_df, x="fe_race_dist" if "fe_race_dist" in dist_df.columns else dist_df.columns[0], y=y, title="거리별 성능")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
        else:
            st.info("거리별 오류 분석 파일 없음")
    with c4:
        if cls_df is not None and not cls_df.empty:
            y = "hit_ratio" if "hit_ratio" in cls_df.columns else "avg_correct" if "avg_correct" in cls_df.columns else None
            if y:
                fig = px.bar(cls_df, x="cndRaceClas" if "cndRaceClas" in cls_df.columns else cls_df.columns[0], y=y, title="등급별 성능")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(cls_df, use_container_width=True, hide_index=True)
        else:
            st.info("등급별 오류 분석 파일 없음")
    st.caption("※ 표본 수가 작을 경우 통계적 변동성이 큽니다.")
