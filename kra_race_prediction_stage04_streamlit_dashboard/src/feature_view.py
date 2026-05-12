"""예측 근거 및 변수 중요도 View."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from . import utils


def _fmt_value(value, suffix: str = "") -> str:
    if pd.isna(value):
        return "데이터 없음"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def render_feature_reasoning(selected_horse_row: pd.DataFrame, feature_importance: pd.DataFrame | None = None) -> None:
    st.subheader("💡 예측 근거/피처 설명")
    if selected_horse_row is None or selected_horse_row.empty:
        st.info("분석할 말을 선택하면 예측 확률, 경주 내 상대 지표, 누적 지표를 표시합니다.")
        return

    s = selected_horse_row.iloc[0]
    prob = s.get("pred_top3_prob", s.get("top3_prob", None))
    if pd.notna(prob) and prob <= 1:
        prob_text = f"{prob * 100:.2f}%"
    elif pd.notna(prob):
        prob_text = f"{prob:.2f}%"
    else:
        prob_text = "데이터 없음"

    rank = s.get("pred_rank_in_race", s.get("pred_rank", None))
    c1, c2, c3 = st.columns(3)
    c1.metric("Top3 예상 확률", prob_text)
    c2.metric("모델 예상 순위", _fmt_value(rank, "위" if pd.notna(rank) else ""))
    c3.metric("출전번호/마명", f"{s.get('pthrHrno', '-')}/{s.get('pthrHrnm', '-')}")

    st.markdown("#### 경주 내 상대 지표")
    relative_cols = [
        "horse_avg_rank_rank_in_race",
        "jockey_top3_rate_rank_in_race",
        "jockey_winrate_rank_in_race",
        "trainer_winrate_rank_in_race",
        "rating_zscore_in_race",
        "weight_zscore_in_race",
    ]
    rel_rows = [{"지표": utils.translate_term(col), "값": _fmt_value(s.get(col))} for col in relative_cols]
    st.dataframe(pd.DataFrame(rel_rows), use_container_width=True, hide_index=True)

    st.markdown("#### 말/기수/조교사 누적 지표")
    cumulative_cols = [
        "fe_horse_cum_avg_rk",
        "fe_horse_cum_win_rate",
        "fe_horse_race_count",
        "fe_jcky_cum_top3_rate",
        "fe_jcky_cum_win_rate",
        "fe_trar_cum_win_rate",
    ]
    cum_rows = [{"지표": utils.translate_term(col), "값": _fmt_value(s.get(col))} for col in cumulative_cols]
    st.dataframe(pd.DataFrame(cum_rows), use_container_width=True, hide_index=True)

    if feature_importance is not None and not feature_importance.empty:
        feature_col = "feature" if "feature" in feature_importance.columns else feature_importance.columns[0]
        important_features = [f for f in feature_importance[feature_col].head(10).astype(str).tolist() if f in selected_horse_row.columns]
        if important_features:
            st.markdown("#### 변수 중요도 Top 피처 중 해당 말의 값")
            rows = [
                {"피처": f, "설명": utils.FEATURE_DESCRIPTION_MAP.get(f, "모델 입력 변수"), "값": _fmt_value(s.get(f))}
                for f in important_features
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.info("SHAP 개별 기여도는 포함하지 않았으며, 현재 화면은 변수 중요도와 주요 상대 지표 기반의 설명입니다.")


def render_feature_importance(fi_df: pd.DataFrame, source_label: str = "v2") -> None:
    st.subheader("🔍 변수 중요도")
    if fi_df is None or fi_df.empty:
        st.warning("변수 중요도 파일이 없어 이 탭의 상세 표와 차트를 표시할 수 없습니다.")
        return

    work = fi_df.copy()
    if "feature" not in work.columns:
        work = work.rename(columns={work.columns[0]: "feature"})
    if "importance" not in work.columns:
        numeric_cols = work.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            work = work.rename(columns={numeric_cols[0]: "importance"})
        else:
            st.error("변수 중요도 파일에서 importance 숫자 컬럼을 찾지 못했습니다.")
            return

    work["importance"] = pd.to_numeric(work["importance"], errors="coerce")
    top20 = work.sort_values("importance", ascending=False).head(20).copy()
    top20["쉬운 설명"] = top20["feature"].map(lambda x: utils.FEATURE_DESCRIPTION_MAP.get(x, "모델의 세부 분석 변수"))
    st.caption(f"사용 파일 우선순위: {source_label}")
    st.dataframe(top20[["feature", "쉬운 설명", "importance"]].rename(columns={"feature": "변수명", "importance": "중요도"}), use_container_width=True, hide_index=True)

    fig = px.bar(top20.sort_values("importance"), x="importance", y="feature", orientation="h", title="변수 중요도 Top 20")
    st.plotly_chart(fig, use_container_width=True)
