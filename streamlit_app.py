# -*- coding: utf-8 -*-
"""
streamlit_app.py - 서울 경마 순위 예측 플랫폼 프로토타입
실행: streamlit run streamlit_app.py

⚠️ 이 앱은 과거 실적 데이터(2025-03 ~ 2026-03)를 기반으로 한
   예측 프로토타입입니다. 실제 베팅 또는 투자 판단 책임은 사용자에게 있습니다.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# ── 페이지 기본 설정 ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="서울 경마 순위 예측 플랫폼",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 한글 폰트 ─────────────────────────────────────────────────────────────
def setup_font():
    font_candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]
    available = {f.name for f in fm.fontManager.ttflist}
    for fnt in font_candidates:
        if fnt in available:
            matplotlib.rc("font", family=fnt)
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

setup_font()

# ── 경로 설정 ─────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent
SRC_DIR        = ROOT / "src"
REPORTS_TABLES = ROOT / "reports" / "GPT" / "outputs" / "tables"
REPORTS_CHARTS = ROOT / "reports" / "GPT" / "outputs" / "charts"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "models"

# ── CSS 스타일 ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700;
        color: #1a1a2e; margin-bottom: 0;
    }
    .subtitle { font-size: 0.95rem; color: #555; margin-top: 4px; }
    .warning-box {
        background: #fff3cd; border-left: 4px solid #ffc107;
        padding: 10px 16px; border-radius: 4px;
        font-size: 0.85rem; color: #856404; margin-bottom: 12px;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 8px;
        padding: 14px; text-align: center;
        border: 1px solid #dee2e6;
    }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #2c3e50; }
    .metric-label { font-size: 0.75rem; color: #6c757d; margin-top: 4px; }
    .rank-1 { background: #fff9c4; font-weight: bold; }
    .rank-top3 { background: #e8f5e9; }
    .badge-noOdds {
        background: #2196F3; color: white;
        padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;
    }
    .badge-withOdds {
        background: #FF9800; color: white;
        padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# ── 주의문구 ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="warning-box">
⚠️ <strong>주의:</strong>
이 앱은 <strong>과거 실적 데이터(2025-03 ~ 2026-03)를 기반으로 한 예측 프로토타입</strong>이며,
<strong>과거 경주 리플레이 기반 예측 데모</strong>입니다.
실제 미래 경주 예측이 아닙니다. 실제 베팅 또는 투자 판단 책임은 전적으로 사용자에게 있으며,
이 플랫폼의 제공자는 어떠한 손실에도 책임지지 않습니다.
</div>
""", unsafe_allow_html=True)

# ── 메인 타이틀 ───────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏇 서울 경마 순위 예측 플랫폼</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">KRA 서울 경마장 · 과거 경주 리플레이 기반 예측 데모 · 2025-03 ~ 2026-03</div>', unsafe_allow_html=True)
st.divider()

# ── 사이드바 ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    mode = st.radio(
        "모델 선택",
        options=["no_odds", "with_odds"],
        format_func=lambda x: "📊 No-Odds (순수 데이터)" if x == "no_odds" else "💰 With-Odds (배당 포함)",
        help="No-Odds: 배당 정보 없이 순수 성적 데이터만 사용\nWith-Odds: 시장 배당 정보 포함"
    )
    st.divider()
    page = st.radio(
        "화면 선택",
        options=["🏆 메인 예측", "🔬 모델 진단", "📋 데이터 점검"],
    )

# ── 데이터 로드 함수 ──────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_predictions(sel_mode: str):
    """예측 결과 캐시 로드"""
    candidates = [
        DATA_PROCESSED / f"final_predictions_{sel_mode}.csv",
        DATA_PROCESSED / "final_predictions.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
            df["rcDate"] = df["rcDate"].astype(str)
            df["rcNo"]   = pd.to_numeric(df["rcNo"], errors="coerce")
            return df
    return None


@st.cache_data
def load_table(fname: str):
    p = REPORTS_TABLES / fname
    if p.exists():
        return pd.read_csv(p, encoding="utf-8-sig")
    return None


@st.cache_data
def load_chart_bytes(fname: str):
    p = REPORTS_CHARTS / fname
    if p.exists():
        return open(p, "rb").read()
    return None


# ── 근거 텍스트 생성 ──────────────────────────────────────────────────────
def get_reasons(row: pd.Series, sel_mode: str) -> str:
    reasons = []
    feat_checks = [
        ("hr_recent3_avg_rank",  lambda v: f"최근3전 평균 {v:.1f}위" if v <= 4 else None),
        ("hr_recent3_top3_rate", lambda v: f"최근3전 Top3율 {v*100:.0f}%" if v >= 0.5 else None),
        ("jk_90d_win_rate",      lambda v: f"기수 90일 승률 {v*100:.1f}%" if v >= 0.12 else None),
        ("hr_jk_cum_win_rate",   lambda v: f"말-기수 조합 승률 {v*100:.1f}%" if v >= 0.2 else None),
        ("tr_30d_win_rate",      lambda v: f"조교사 30일 승률 {v*100:.1f}%" if v >= 0.12 else None),
        ("hr_cum_win_rate",      lambda v: f"누적 승률 {v*100:.1f}%" if v >= 0.18 else None),
        ("hr_prev1_rank",        lambda v: f"직전 경주 {int(v)}위" if v <= 2 else None),
        ("hr_rank_trend",        lambda v: "성적 상승 추세" if v < -1 else None),
    ]
    if sel_mode == "with_odds":
        feat_checks.append(("winOdds", lambda v: f"배당 {v:.1f}배 (우위)" if v <= 4 else None))

    for feat, fn in feat_checks:
        if feat in row.index and pd.notna(row[feat]):
            txt = fn(float(row[feat]))
            if txt:
                reasons.append(txt)
        if len(reasons) >= 3:
            break

    if not reasons:
        reasons.append("데이터 부족")
    return " / ".join(reasons[:3])


# ════════════════════════════════════════════════════════════════
# 화면 1: 메인 예측 화면
# ════════════════════════════════════════════════════════════════
if page == "🏆 메인 예측":
    st.subheader("🏆 경주별 순위 예측")

    comp_df = load_table("13_model_comparison.csv")
    if comp_df is not None:
        target_model = "Ensemble_WithOdds" if mode == "with_odds" else "Ensemble_NoOdds"
        model_row = comp_df[comp_df["model"] == target_model]
        if not model_row.empty:
            win_hr = pd.to_numeric(model_row.iloc[0]["winner_hit_rate"], errors="coerce")
            top3_hr = pd.to_numeric(model_row.iloc[0]["top3_hit_rate"], errors="coerce")
            st.info(
                f"💡 **현재 파이프라인의 전반적인 예측 성공률 (모델: {target_model})**\n\n"
                f"• **1위 적중률(Winner Hit Rate):** {win_hr * 100:.1f}%\n"
                f"• **3위 내 적중률(Top3 Hit Rate):** {top3_hr * 100:.1f}%\n\n"
                f"*경마 산업 특유의 높은 불확실성과 이변, 그리고 모델 학습 시 가장 보수적인 기준(미래 정보 완전 차단)을 적용함에 따라 대다수 단일 경주의 예측은 실제 상위권 결과와 어긋날 확률이 높습니다. 본 화면은 특정 경주의 리플레이 및 지표 보조용으로 참고하십시오.*"
            )

    preds = load_predictions(mode)

    if preds is None:
        st.warning("⚠️ 예측 결과 파일이 없습니다. 먼저 파이프라인을 실행하세요.")
        st.code("python run_pipeline.py", language="bash")
        st.stop()

    # 날짜 / 경주번호 선택
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        dates = sorted(preds["rcDate"].unique(), reverse=True)
        sel_date = st.selectbox("📅 경주일 선택", dates)

    with col2:
        day_df = preds[preds["rcDate"] == sel_date]
        races  = sorted(day_df["rcNo"].dropna().unique().astype(int).tolist())
        if not races:
            st.warning("해당 날짜에 경주 정보가 없습니다.")
            st.stop()
        sel_race = st.selectbox("🏁 경주번호", races)

    with col3:
        show_actual = st.toggle("실제 결과 표시", value=True)

    # 경주 데이터 필터
    race_df = day_df[day_df["rcNo"] == sel_race].copy()
    if race_df.empty:
        st.info("해당 경주 예측 데이터가 없습니다.")
        st.stop()

    # 예상 순위 정렬
    if "pred_rank" not in race_df.columns and "final_score" in race_df.columns:
        race_df["pred_rank"] = race_df["final_score"].rank(ascending=False).astype(int)
    race_df = race_df.sort_values("pred_rank")

    # 경주 기본 정보
    st.markdown(f"**{sel_date} | {sel_race}경주**  |  출전두수: **{len(race_df)}두**")
    badge = '<span class="badge-noOdds">No-Odds</span>' if mode == "no_odds" \
            else '<span class="badge-withOdds">With-Odds</span>'
    st.markdown(f"모델: {badge}", unsafe_allow_html=True)

    # 요약 지표
    win_pred  = race_df.iloc[0]
    top3_pred = race_df.iloc[:3]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🥇 예상 1위", win_pred.get("hrName", "N/A"))
    with c2:
        win_p = win_pred.get("win_prob", None)
        st.metric("1위 확률", f"{win_p:.1%}" if pd.notna(win_p) else "N/A")
    with c3:
        top3_p = win_pred.get("top3_prob", None)
        st.metric("Top3 확률", f"{top3_p:.1%}" if pd.notna(top3_p) else "N/A")
    with c4:
        score = win_pred.get("final_score", None)
        st.metric("종합 Score", f"{score:.3f}" if pd.notna(score) else "N/A")

    st.divider()

    # 예측 테이블 구성
    show_cols = {
        "chulNo":     "출전번호",
        "hrName":     "말명",
        "jkName":     "기수",
        "trName":     "조교사",
        "pred_rank":  "예상순위",
        "final_score":"종합Score",
        "win_prob":   "1위확률",
        "top3_prob":  "Top3확률",
    }
    if show_actual and "ord" in race_df.columns:
        show_cols["ord"] = "실제순위"
    if mode == "with_odds" and "winOdds" in race_df.columns:
        show_cols["winOdds"] = "배당(단승)"

    avail = {k: v for k, v in show_cols.items() if k in race_df.columns}
    display_df = race_df[list(avail.keys())].rename(columns=avail).copy()

    # 근거 컬럼
    display_df["예측 근거"] = race_df.apply(lambda r: get_reasons(r, mode), axis=1).values

    # 소수점 포맷
    for col in ["종합Score", "1위확률", "Top3확률"]:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(3)

    st.dataframe(
        display_df.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "예상순위": st.column_config.NumberColumn(format="%d위"),
            "1위확률":  st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
            "Top3확률": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
        }
    )

    # 실제 결과와 비교 (show_actual = True 일 때)
    if show_actual and "ord" in race_df.columns and "실제순위" in display_df.columns:
        correct_win  = (race_df["pred_rank"] == 1).any() and (race_df[race_df["pred_rank"]==1]["ord"] == 1).any()
        correct_top3 = (race_df[race_df["pred_rank"] <= 3]["ord"] <= 3).any()
        col_a, col_b = st.columns(2)
        with col_a:
            if correct_win:
                st.success("✅ 1위 예측 적중!")
            else:
                actual_winner = race_df[race_df["ord"]==1]["hrName"].values
                st.error(f"❌ 1위 예측 빗나감 (실제: {actual_winner[0] if len(actual_winner) else 'N/A'})")
        with col_b:
            if correct_top3:
                st.success("✅ Top3 예측 1마리 이상 적중")
            else:
                st.warning("⚠️ Top3 예측 전부 빗나감")


# ════════════════════════════════════════════════════════════════
# 화면 2: 모델 진단 화면
# ════════════════════════════════════════════════════════════════
elif page == "🔬 모델 진단":
    st.subheader("🔬 모델 진단")

    # 모델 비교표
    st.markdown("#### 📊 모델 성능 비교 (Test Set)")
    comp = load_table("13_model_comparison.csv")
    if comp is not None:
        st.dataframe(comp, use_container_width=True, hide_index=True)
    else:
        bl = load_table("10_baseline_results.csv")
        if bl is not None:
            st.info("통합 비교표 없음. 베이스라인 결과만 표시합니다.")
            st.dataframe(bl, use_container_width=True, hide_index=True)
        else:
            st.warning("모델 비교표 없음. evaluate.py를 실행하세요.")

    st.divider()

    # 차트 3열 레이아웃
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("##### 피처 중요도 (No-Odds)")
        img = load_chart_bytes("10_feature_importance_no_odds.png")
        if img:
            st.image(img, use_container_width=True)
        else:
            st.info("평가 파이프라인 실행 후 생성됩니다.")

    with c2:
        st.markdown("##### 피처 중요도 (With-Odds)")
        img = load_chart_bytes("11_feature_importance_with_odds.png")
        if img:
            st.image(img, use_container_width=True)
        else:
            st.info("평가 파이프라인 실행 후 생성됩니다.")

    with c3:
        st.markdown("##### Calibration 진단")
        img = load_chart_bytes("12_calibration_plot.png")
        if img:
            st.image(img, use_container_width=True)
        else:
            st.info("보정 파이프라인 실행 후 생성됩니다.")

    # 월별 성능
    st.divider()
    st.markdown("#### 📅 월별 성능 안정성")
    monthly = load_table("14_monthly_performance.csv")
    if monthly is not None:
        st.dataframe(monthly, use_container_width=True, hide_index=True)

        # 월별 차트
        fig, ax = plt.subplots(figsize=(10, 4))
        x = range(len(monthly))
        if "winner_hit_rate" in monthly.columns:
            ax.plot(x, pd.to_numeric(monthly["winner_hit_rate"], errors="coerce"),
                    marker="o", label="Winner Hit Rate", color="#2E86AB", linewidth=2)
        if "top3_hit_rate" in monthly.columns:
            ax.plot(x, pd.to_numeric(monthly["top3_hit_rate"], errors="coerce"),
                    marker="s", label="Top3 Hit Rate", color="#A23B72", linewidth=2)
        ax.set_xticks(list(x))
        ax.set_xticklabels(monthly["year_month"].values, rotation=30, ha="right")
        ax.set_title("월별 Hit Rate 추이", fontsize=12, fontweight="bold")
        ax.set_ylabel("Hit Rate")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("월별 성능 데이터 없음.")

    # 모델 비교 차트
    st.divider()
    st.markdown("#### 🏅 Model Hit Rate 비교")
    img = load_chart_bytes("13_model_comparison.png")
    if img:
        st.image(img, use_container_width=True)

    # 누설 제거 로그 요약
    st.divider()
    st.markdown("#### 🔒 누설 컬럼 제거 로그")
    col_class = load_table("01_column_classification.csv")
    if col_class is not None:
        leak_df = col_class[col_class["group"].str.contains("누설|제거", na=False)]
        st.dataframe(
            leak_df[["column_name", "group", "reason"]],
            use_container_width=True, hide_index=True
        )
        st.caption(f"총 {len(leak_df)}개 컬럼 제거/제한")


# ════════════════════════════════════════════════════════════════
# 화면 3: 데이터 점검 화면
# ════════════════════════════════════════════════════════════════
elif page == "📋 데이터 점검":
    st.subheader("📋 데이터 점검")

    # 기본 요약
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("데이터 기간", "2025-03 ~ 2026-03")
    with c2:
        st.metric("총 경주 수", "1,150경주")
    with c3:
        st.metric("총 출전 레코드", "12,206건")
    with c4:
        st.metric("경마장", "서울")

    st.divider()

    # 결측률 상위
    st.markdown("#### 📉 결측률 상위 컬럼 (상위 20)")
    miss_df = load_table("02_missing_rate.csv")
    if miss_df is not None:
        top_miss = miss_df[miss_df["missing_rate_pct"] > 0].nlargest(20, "missing_rate_pct")
        st.dataframe(top_miss[["column", "missing_rate_pct", "missing_count", "dtype"]],
                     use_container_width=True, hide_index=True)
    else:
        st.info("data_audit.py 실행 후 생성됩니다.")

    st.divider()

    # 2열 레이아웃
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🔴 특수코드(ord≥90) 분포")
        special_df = load_table("03_special_code_freq.csv")
        if special_df is not None:
            st.dataframe(special_df, use_container_width=True, hide_index=True)
            st.caption("92=실격?, 93=취소?, 94=부상/중도기권?, 95=출전취소? (정확한 의미 정의 확인 필요)")

            # 특수코드 차트
            img = load_chart_bytes("03_special_code_dist.png")
            if img:
                st.image(img, use_container_width=True)
        else:
            st.info("data_audit.py 실행 후 생성됩니다.")

    with col_b:
        st.markdown("#### 📊 경주당 출전두수 분포")
        img = load_chart_bytes("01_field_size_dist.png")
        if img:
            st.image(img, use_container_width=True)
        else:
            st.info("data_audit.py 실행 후 생성됩니다.")

    st.divider()

    # 피처 현황
    st.markdown("#### 🔧 피처 현황")
    feat_path_no   = Path("./artifacts/feature_list_no_odds.json")
    feat_path_with = Path("./artifacts/feature_list_with_odds.json")

    col_x, col_y = st.columns(2)
    with col_x:
        if feat_path_no.exists():
            import json
            with open(feat_path_no) as f:
                feat_no = json.load(f)
            st.metric("No-Odds 피처 수", len(feat_no.get("features", [])))
            with st.expander("피처 목록 보기"):
                st.write(feat_no.get("features", []))
        else:
            st.info("feature_engineering.py 실행 후 생성됩니다.")

    with col_y:
        if feat_path_with.exists():
            import json
            with open(feat_path_with) as f:
                feat_w = json.load(f)
            st.metric("With-Odds 피처 수", len(feat_w.get("features", [])))
            with st.expander("피처 목록 보기"):
                st.write(feat_w.get("features", []))
        else:
            st.info("feature_engineering.py 실행 후 생성됩니다.")

    st.divider()

    # 의미 불명확 컬럼
    st.markdown("#### ❓ 정의 불명확 컬럼 (처리 보류)")
    undefined_cols = [
        {"컬럼명": "chulYn",  "추정 의미": "출전 여부 코드", "처리": "제거(사용 보류)"},
        {"컬럼명": "differ",  "추정 의미": "불명확",         "처리": "제거(사용 보류)"},
        {"컬럼명": "owCloth", "추정 의미": "마주복 코드?",   "처리": "제거(사용 보류)"},
        {"컬럼명": "meet_nm", "추정 의미": "경마장명(100%결측)","처리": "제거"},
        {"컬럼명": "df",      "추정 의미": "불명확",         "처리": "제거(사용 보류)"},
    ]
    st.dataframe(pd.DataFrame(undefined_cols), use_container_width=True, hide_index=True)

    st.divider()

    # 컬럼 분류표
    st.markdown("#### 📑 컬럼 위험도 분류표")
    col_class = load_table("01_column_classification.csv")
    if col_class is not None:
        group_filter = st.multiselect(
            "그룹 필터",
            options=col_class["group"].unique().tolist(),
            default=col_class["group"].unique().tolist()
        )
        filtered = col_class[col_class["group"].isin(group_filter)]
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.caption(f"총 {len(filtered)}개 컬럼 표시 / 전체 {len(col_class)}개")
    else:
        st.info("data_audit.py 실행 후 생성됩니다.")

    # 월별 경주 분포
    st.divider()
    st.markdown("#### 📅 월별 경주 분포")
    race_stats = load_table("04_race_stats.csv")
    if race_stats is not None:
        st.dataframe(race_stats, use_container_width=True, hide_index=True)
        img = load_chart_bytes("02_monthly_race_count.png")
        if img:
            st.image(img, use_container_width=True)
    else:
        st.info("data_audit.py 실행 후 생성됩니다.")
