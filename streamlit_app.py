import streamlit as st
import pandas as pd
import os
import warnings

# ──────────────────────────────────────────────
# 0. 배포 환경 경로 보정 불필요 (Root 실행)
# 프로젝트 최상단(root)에서 실행되므로 경로 수정을 전부 들어냈습니다.
# ──────────────────────────────────────────────

from kra_race_prediction_stage04_streamlit_dashboard.src import utils
from kra_race_prediction_stage04_streamlit_dashboard.src.data_loader import load_all_datasets
from kra_race_prediction_stage04_streamlit_dashboard.src.prediction_service import load_model, predict_top3
from kra_race_prediction_stage04_streamlit_dashboard.src.feature_view import render_feature_reasoning, render_feature_importance
from kra_race_prediction_stage04_streamlit_dashboard.src.evaluation_view import render_rule_comparison, render_model_performance, render_error_analysis

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. 앱 기본 구성 및 상태 로드
# ──────────────────────────────────────────────
st.set_page_config(page_title="KRA Top3 Prediction Prototype", layout="wide", page_icon="🐎")

utils.render_warning_disclaimer()

st.title("🏇 KRA 경주마 Top3 예측 대시보드 (Prototype)")

# 데이터 & 모델 캐싱 로드
data = load_all_datasets()
lgb_model = load_model()

if data.get("modeling_ready") is None or lgb_model is None:
    st.error("치명적 오류: 핵심 필수 데이터 셋(또는 모델)을 불러오지 못했습니다. 경로를 확인해주세요.")
    st.stop()

# ──────────────────────────────────────────────
# 2. 사이드바 - 모드 및 필터 선택
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 사용자 메뉴")
    
    # 평가 검증 모드 vs 실운영 모드 vs 실시간 미래 예측 토글
    app_mode = st.radio(
        "사용 모드 선택",
        options=["과거 검증 모드", "운영 예측 모드(시뮬레이션)", "🚀 실시간 미래 경주 예측"],
        help="과거 검증: 이미 결과가 나온 데이터의 정답과 비교 / 운영 예측: 정답 블라인드 / 미래 예측: Stage 05 파이프라인 결과 출력"
    )
    is_past_mode = (app_mode == "과거 검증 모드")
    is_future_mode = (app_mode == "🚀 실시간 미래 경주 예측")
    
    st.divider()
    
    if is_future_mode:
        if os.path.exists(utils.PATH_NEXT_PRED):
            future_df = pd.read_csv(utils.PATH_NEXT_PRED, encoding="utf-8-sig")
            available_races = sorted(list(future_df['race_id'].unique()))
            selected_race = st.selectbox("미래 경주(Race ID) 선택", options=available_races)
            race_display_df = future_df[future_df['race_id'] == selected_race].copy()
            st.success(f"미래 예측 데이터 로드 완료: {len(race_display_df)}두")
        else:
            st.error("미래 경주 예측 결과 파일이 없습니다. 파이프라인을 먼저 실행해주세요.")
            st.stop()
    else:
        mod_df = data["modeling_ready"]
        
        # 시간 기반으로 정렬된 일자 가져오기
        available_dates = sorted(list(mod_df['schdRaceDt'].unique()), reverse=True)
        selected_date = st.selectbox("경주일자 선택", options=available_dates)
        
        date_df = mod_df[mod_df['schdRaceDt'] == selected_date]
        available_races = sorted(list(date_df['race_id'].unique()))
        selected_race = st.selectbox("경주(Race ID) 선택", options=available_races)

        # 선택된 특정 레이스의 출전마 DF (모델 피처 포함)
        race_full_df = date_df[date_df['race_id'] == selected_race].copy()
        
        st.info(f"선택 데이터: {selected_date} / {selected_race}\n출전: {len(race_full_df)}두")

if is_future_mode:
    # 미래 모드에서는 이미 산출된 결과(race_display_df)를 사용
    pred_race_df = race_display_df.rename(columns={
        'top3_prob': 'pred_top3_prob',
        'pred_rank': 'pred_rank_in_race'
    })
    # is_top3를 상위 3마리에 대해 마킹
    pred_race_df['pred_is_top3'] = (pred_race_df['pred_rank_in_race'] <= 3).astype(int)
else:
    # 기존 on-the-fly 추론
    pred_race_df = predict_top3(race_full_df, lgb_model)
    if pred_race_df is None or pred_race_df.empty:
        st.error("현재 선택된 경주를 예측할 수 없습니다.")
        st.stop()

# ──────────────────────────────────────────────
# 4. 메인 탭 구현 
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📂 1. 프로젝트 개요", 
    "🎯 2. 경주 예측 결과",
    "💡 3. 예측 근거 설명",
    "⚖️ 4. 단순 규칙 비교",
    "📊 5. 모델 성능 요약",
    "🔍 6. 변수 중요도",
    "🎯 7. 오류 분석"
])

with tab1:
    st.header("프로젝트 개요")
    st.write("본 시스템은 한국마사회(KRA) 서울 경주 데이터를 기반으로 경마 정보를 학습하여 경주마의 순위 3위(Top3) 안착 가능성을 인공지능이 계산하는 **프로토타입 애플리케이션**입니다.")
    st.markdown("""
    * **모델 방식**: LightGBM Binary Classifier (경주 내 상위 3머리 선발)
    * **활용 정보**: 경주 전 이미 공개되는 말, 기수, 조교사의 과거 누적 데이터 및 배정된 레이팅/부담중량 정보
    """)
    
    st.divider()
    st.subheader("📋 LightGBM 모델 Test-Set 성능 요약 (가이드라인)")
    
    # 하드코딩 또는 동적 주입
    # 요구사항에 명시된 수치 표기
    perf_data = {
        "항목": ["Test 경주 수", "LightGBM Precision@3", "LightGBM Hit@3", "무작위 기준선", "말 평균순위 규칙 Precision@3"],
        "값": ["520", "51.60%", "93.27%", "28.3%", "46.03%"]
    }
    st.table(pd.DataFrame(perf_data))
    
with tab2:
    header_text = f"[{selected_race}] 미래 경주 예측 결과" if is_future_mode else f"[{selected_date}] Race: {selected_race} 예측 결과"
    st.header(header_text)
    
    # 표시용 데이터프레임 가공
    disp_df = pred_race_df.copy()
    
    # 필요한 표시용 컬럼 생성 (candidate_cols에서 참조하는 이름들)
    if 'pred_rank_in_race' in disp_df.columns:
        disp_df['모델 예상 순위'] = disp_df['pred_rank_in_race']
    if 'pred_top3_prob' in disp_df.columns:
        disp_df['Top3 예상 확률(%)'] = (disp_df['pred_top3_prob'] * 100).round(1)
    if 'pred_is_top3' in disp_df.columns:
        disp_df['모델 선택(Top3)'] = disp_df['pred_is_top3'].apply(lambda x: "✅ 선택" if x == 1 else "")

    # 미래 모드와 과거 검증 모드에 따른 표시 컬럼 분리
    if is_future_mode:
        candidate_cols = [
            "모델 예상 순위", "pthrGtno", "pthrHrno", "pthrHrnm", "hrmJckyNm", "hrmTrarNm", 
            "pthrRatg", "pthrBurdWgt", "fe_horse_cum_avg_rk", "fe_jcky_cum_top3_rate", 
            "fe_trar_cum_win_rate", "Top3 예상 확률(%)", "모델 선택(Top3)"
        ]
        st.info("미래 경주 예측 모드입니다. 실제 결과가 없는 데이터이므로 정답 컬럼은 표시하지 않습니다.")
    else:
        candidate_cols = [
            "모델 예상 순위", "pthrHrno", "pthrRatg", "pthrBurdWgt", 
            "fe_horse_cum_avg_rk", "fe_jcky_cum_top3_rate", "fe_trar_cum_win_rate", 
            "Top3 예상 확률(%)", "모델 선택(Top3)"
        ]
        if 'target_is_top3' in disp_df.columns:
            disp_df['👑 실제 대상(Top3)'] = disp_df['target_is_top3'].apply(lambda x: "✔️ 적중" if x==1 else "")
            candidate_cols.append('👑 실제 대상(Top3)')
        else:
            st.info("운영 예측 모드입니다. 경주 결과(정답)는 블라인드 처리되었습니다.")

    # 존재하는 컬럼만 선별하여 표시 (KeyError 방지)
    view_cols = [c for c in candidate_cols if c in disp_df.columns]
    missing_display_cols = [c for c in candidate_cols if c not in disp_df.columns]
    
    if missing_display_cols:
        st.warning(f"일부 표시 컬럼이 데이터에 없어 제외했습니다: {missing_display_cols}")
        
    final_view = disp_df[view_cols].sort_values("모델 예상 순위")
    final_view = utils.apply_friendly_columns(final_view)
    
    st.dataframe(final_view, use_container_width=True, hide_index=True)

with tab3:
    st.header("예측 근거 (Feature Justification)")
    st.markdown("특정 말의 세부 역량(상대적 우위 등)을 조회하여 모델의 판단 사유를 이해합니다.")
    
    sorted_df = pred_race_df.sort_values("pred_top3_prob", ascending=False)
    horse_opts = sorted_df['pthrHrno'].astype(str).tolist()
    
    if horse_opts:
        sel_hrno = st.selectbox("분석 대상 말 출전번호 선택", options=horse_opts)
        sel_row = pred_race_df[pred_race_df['pthrHrno'].astype(str) == sel_hrno]
        render_feature_reasoning(sel_row)
    else:
        st.write("해당 경주 정보가 부족합니다.")

with tab4:
    # 룰 엔진 데이터는 data['rule_pred'] 에서 조회 
    rule_df = data.get("rule_pred", pd.DataFrame())
    lgbm_df = data.get("lgbm_pred", pd.DataFrame())
    
    if not isinstance(rule_df, pd.DataFrame) or rule_df.empty:
        st.warning("단순 규칙 파일(baseline_rule_predictions.csv)을 찾을 수 없습니다.")
    else:
        render_rule_comparison(rule_df, lgbm_df, selected_race, is_past_mode)

with tab5:
    render_model_performance(data.get("eval_table", pd.DataFrame()))

with tab6:
    render_feature_importance(data.get("feature_importances", pd.DataFrame()))

with tab7:
    render_error_analysis(
        data.get("err_good", pd.DataFrame()),
        data.get("err_bad", pd.DataFrame()),
        data.get("err_dist", pd.DataFrame()),
        data.get("err_cls", pd.DataFrame())
    )
