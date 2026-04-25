# -*- coding: utf-8 -*-
"""
config.py - 프로젝트 전역 설정 중앙 관리
모든 경로, 컬럼 목록, 분할 기준을 여기서 관리한다.
"""

from pathlib import Path

# ── 기본 경로 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW        = ROOT / "data" / "team_source"
DATA_PROCESSED  = ROOT / "data" / "processed"
MODELS_DIR      = ROOT / "models"
REPORTS_TABLES  = ROOT / "reports" / "GPT" / "outputs" / "tables"
REPORTS_CHARTS  = ROOT / "reports" / "GPT" / "outputs" / "charts"
DOCS_DIR        = ROOT / "Docs"
LOGS_DIR        = ROOT / "logs"
ARTIFACTS_DIR   = ROOT / "artifacts"

# 주요 파일 경로
MAIN_CSV        = DATA_PROCESSED / "merged_team_source.csv"
RACE_META_CSV   = DATA_RAW / "4_경주기록정보.csv"   # rcDist, weather, track 등 포함

# ── 데이터 분할 기준 날짜 (yyyymmdd 정수) ─────────────────────────────────
TRAIN_END_DATE  = 20251231   # Train: ~ 2025-12-31
VALID_END_DATE  = 20260131   # Valid: 2026-01-01 ~ 2026-01-31
# Test: 2026-02-01 ~

# ── 재현성 ──────────────────────────────────────────────────────────────────
SEED = 42

# ── 누설 컬럼 (학습에서 무조건 제거) ─────────────────────────────────────
LEAKAGE_COLS = [
    # 결과/사후 정보
    "ord", "ordBigo", "diffUnit", "diffUnit_numeric", "rcTime",
    "is_winner", "is_place", "stOrd",
    # 경주 중/후 구간 기록
    "buG1fAccTime", "buG1fOrd", "buG2fAccTime", "buG2fOrd",
    "buG3fAccTime", "buG3fOrd", "buG4fAccTime", "buG4fOrd",
    "buG6fAccTime", "buG6fOrd", "buG8fAccTime", "buG8fOrd",
    "buS1fAccTime", "buS1fOrd", "buS1fTime",
    "bu_10_8fTime", "bu_1fGTime", "bu_2fGTime", "bu_3fGTime",
    "bu_4_2fTime", "bu_6_4fTime", "bu_8_6fTime",
    "je_1cTime", "je_2cTime", "je_3cTime", "je_4cTime",
    "se_1cAccTime", "se_2cAccTime", "se_3cAccTime", "se_4cAccTime",
    "sj_1cOrd", "sj_2cOrd", "sj_3cOrd", "sj_4cOrd",
]

# _ana 계열 (전부 100% 결측 → 제거)
ANA_COLS_PATTERN = "_ana"

# ── With-Odds 전용 컬럼 (No-Odds 모델에서 제거) ───────────────────────────
ODDS_COLS = [
    "winOdds", "plcOdds", "win", "plc", "implied_rank_from_win_odds",
]

# ── 정의 불명확 컬럼 (사용 보류) ─────────────────────────────────────────
UNDEFINED_COLS = [
    "chulYn", "differ", "owCloth", "meet_nm", "df",
    "track_moisture_pct",   # 100% 결측
]

# ── ID 컬럼 (과적합 방지 - 직접 feature로 사용 금지) ─────────────────────
ID_COLS = [
    "entry_id", "race_id", "hrNo", "jkNo", "trNo", "owNo",
    "hrName", "jkName", "trName", "owName",
    "rcDate", "rcDate_dt", "rcNo", "birthday",
]

# ── 범주형 컬럼 ─────────────────────────────────────────────────────────────
CAT_COLS = [
    "sex", "name",          # 성별, 생산국가
    "rank_enc",             # 등급 인코딩
    "track_enc",            # 주로상태 인코딩
    "weather_enc",          # 날씨 인코딩
    "dist_group",           # 거리군
    "gate_zone",            # 내/중/외측 게이트 구간
    "has_tool",             # 장구 사용 여부
]

# ── 정적 피처 (출전 당일 알 수 있는 정보) ───────────────────────────────────
STATIC_FEATURES = [
    "age", "sex", "wgBudam", "wgJk",
    "chulNo", "chulNo_relative",     # 출전번호, 상대 위치
    "gate_zone",
    "field_size",                    # 경주당 출전두수
    "tool_count", "has_tool",        # 장구 피처
    "name",                          # 생산국가 계열
    "hrRating",                      # 레이팅
    # 병합 후 추가
    "rcDist", "dist_group",
    "track_enc", "weather_enc",
    "rank_enc", "prizeCond_enc",
]

# ── 말 기준 롤링 피처 ─────────────────────────────────────────────────────
HORSE_FEATURES = [
    "hr_prev1_rank",
    "hr_recent3_avg_rank", "hr_recent5_avg_rank",
    "hr_recent1_win_rate", "hr_recent3_win_rate", "hr_recent5_win_rate",
    "hr_recent1_top3_rate", "hr_recent3_top3_rate", "hr_recent5_top3_rate",
    "hr_recent3_avg_field_size",
    "hr_recent3_avg_weight_delta",
    "hr_recent3_avg_rest_days",
    "hr_last_rest_days",
    "hr_cum_starts", "hr_cum_win_rate", "hr_cum_top3_rate",
    "hr_rank_trend",            # 최근3전 - 이전3전 평균순위 차이
]

# ── 기수 기준 피처 ─────────────────────────────────────────────────────────
JOCKEY_FEATURES = [
    "jk_cum_starts", "jk_cum_win_rate", "jk_cum_top3_rate",
    "jk_30d_win_rate", "jk_30d_top3_rate",
    "jk_90d_win_rate", "jk_90d_top3_rate",
    "jk_recent_avg_rank",
]

# ── 조교사 기준 피처 ───────────────────────────────────────────────────────
TRAINER_FEATURES = [
    "tr_cum_starts", "tr_cum_win_rate", "tr_cum_top3_rate",
    "tr_30d_win_rate", "tr_30d_top3_rate",
    "tr_90d_win_rate", "tr_90d_top3_rate",
    "tr_recent_avg_rank",
]

# ── 마주 기준 피처 ─────────────────────────────────────────────────────────
OWNER_FEATURES = [
    "ow_cum_starts", "ow_cum_win_rate", "ow_cum_top3_rate",
    "ow_30d_win_rate", "ow_90d_win_rate",
    "ow_recent_avg_rank",
]

# ── 조합 피처 ──────────────────────────────────────────────────────────────
COMBO_FEATURES = [
    "hr_jk_cum_starts", "hr_jk_cum_win_rate", "hr_jk_cum_top3_rate",
    "hr_tr_cum_starts", "hr_tr_cum_win_rate", "hr_tr_cum_top3_rate",
]

# ── 최종 No-Odds 피처 목록 (전체 조합) ────────────────────────────────────
NO_ODDS_FEATURES = (
    STATIC_FEATURES
    + HORSE_FEATURES
    + JOCKEY_FEATURES
    + TRAINER_FEATURES
    + OWNER_FEATURES
    + COMBO_FEATURES
)

# ── With-Odds 추가 피처 ────────────────────────────────────────────────────
WITH_ODDS_EXTRA = [
    "winOdds", "plcOdds", "implied_rank_from_win_odds",
]
WITH_ODDS_FEATURES = NO_ODDS_FEATURES + WITH_ODDS_EXTRA

# ── 폴더 자동 생성 ─────────────────────────────────────────────────────────
for _d in [DATA_PROCESSED, MODELS_DIR, REPORTS_TABLES, REPORTS_CHARTS,
           DOCS_DIR, LOGS_DIR, ARTIFACTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
