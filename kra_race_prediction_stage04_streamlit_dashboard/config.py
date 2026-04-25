"""
00_config.py
──────────────────────────────
Stage 03: Top3 Baseline Modeling 공통 환경 설정 및 유틸 함수
"""

import os
import sys
import warnings

# Windows 환경 한글 인코딩 오류 방지
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    try:
        sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    except Exception:
        pass

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. 경로 설정
# ──────────────────────────────────────────────
# 현재 스크립트의 위치: src/
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# 스테이지 루트: kra_race_prediction_stage03_top3_modeling/
STAGE_ROOT = os.path.dirname(SRC_DIR)
# 전체 프로젝트 루트: kra_api227_racedetailresult_project/
PROJECT_ROOT = os.path.dirname(STAGE_ROOT)

# 데이터 경로
DATA_INPUT      = os.path.join(STAGE_ROOT, "data", "input")
DATA_PROCESSED  = os.path.join(STAGE_ROOT, "data", "processed")
DATA_MODELING   = os.path.join(STAGE_ROOT, "data", "modeling")
DATA_PREDICTIONS= os.path.join(STAGE_ROOT, "data", "predictions")

# 출력물 경로
OUT_TABLES      = os.path.join(STAGE_ROOT, "outputs", "tables")
OUT_FIGS        = os.path.join(STAGE_ROOT, "outputs", "figures")
OUT_METRICS     = os.path.join(STAGE_ROOT, "outputs", "metrics")
OUT_DIAGNOSTICS = os.path.join(STAGE_ROOT, "outputs", "diagnostics")

# 모델 및 리포트
MODELS_DIR      = os.path.join(STAGE_ROOT, "models")
REPORTS_DIR     = os.path.join(STAGE_ROOT, "reports")

# 이전 스테이지 연계 입력 (modeling_dataset_top3.csv)
PREV_STAGE      = os.path.join(PROJECT_ROOT, "kra_race_prediction_stage02_predictive_eda")
PREVQ_MODELING  = os.path.join(PREV_STAGE, "data", "modeling_ready", "modeling_dataset_top3.csv")


# ──────────────────────────────────────────────
# 2. 메타데이터 (누수 방지 등)
# ──────────────────────────────────────────────
LEAKAGE_COLS = [
    "rsutRk", "target_rank", "target_is_top3", "rsutRaceRcd", 
    "rsutMargin", "rsutRkAdmny", "rsutRkPurse", "rsutQnlaPrice", 
    "rsutWinPrice", "rsutRkRemk", "rsutRlStrtTim", "rsutStrtTimChgRs"
]

TARGET_COL = "target_is_top3"


# ──────────────────────────────────────────────
# 3. 유틸 함수
# ──────────────────────────────────────────────

def setup_plot():
    """Matplotlib 시각화 공통 환경 설정 (한글 폰트 적용 포함)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 기본 설정 폰트 리스트
    font_candidates = [
        "C:/Windows/Fonts/malgun.ttf", 
        "C:/Windows/Fonts/NanumGothic.ttf",
        "C:/Windows/Fonts/gulim.ttc"
    ]
    
    for fc in font_candidates:
        if os.path.exists(fc):
            fe = fm.FontEntry(fname=fc, name='KoreanFont')
            fm.fontManager.ttflist.insert(0, fe)
            plt.rcParams['font.family'] = 'KoreanFont'
            break

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    return plt

def log(v: str):
    print(f"[INFO] {v}", flush=True)

def log_step(n_step: int, msg: str):
    print(f"\n=============================================", flush=True)
    print(f"  Step {n_step}: {msg}", flush=True)
    print(f"=============================================", flush=True)
