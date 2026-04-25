"""
config_loader.py
─────────────────────
각 스크립트에서 공통 설정을 임포트하기 위한 모듈
src/ 디렉토리 내에 위치하며, 모든 스크립트가 공유하는
경로, 상수, 유틸 함수를 정의한다.
"""

import os
import sys
import warnings

# Windows 콘솔 UTF-8 출력 강제
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    try:
        sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    except Exception:
        pass

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 경로 정의
# ─────────────────────────────────────────────
# 이 파일 위치: .../kra_race_prediction_stage02_predictive_eda/src/
SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
STAGE02_DIR  = os.path.dirname(SRC_DIR)

# 상위 프로젝트 루트 (kra_api227_racedetailresult_project/)
PROJECT_ROOT = os.path.dirname(STAGE02_DIR)

# 원본 데이터 경로
RAW_CSV   = os.path.join(PROJECT_ROOT, "kra_race_prediction_eda",
                          "race_results_seoul_3years_revised.csv")
SCHEMA_XL = os.path.join(PROJECT_ROOT, "kra_race_prediction_eda", "schema_info.xlsx")

# Stage02 내부 경로
DATA_RAW       = os.path.join(STAGE02_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(STAGE02_DIR, "data", "processed")
DATA_MODELING  = os.path.join(STAGE02_DIR, "data", "modeling_ready")
OUT_TABLES     = os.path.join(STAGE02_DIR, "outputs", "tables")
OUT_FIGS       = os.path.join(STAGE02_DIR, "outputs", "figures")
OUT_DIAG       = os.path.join(STAGE02_DIR, "outputs", "diagnostics")
REPORTS        = os.path.join(STAGE02_DIR, "reports")

for _p in [DATA_RAW, DATA_PROCESSED, DATA_MODELING,
           OUT_TABLES, OUT_FIGS, OUT_DIAG, REPORTS]:
    os.makedirs(_p, exist_ok=True)

# ─────────────────────────────────────────────
# 공통 상수
# ─────────────────────────────────────────────

# 누수 컬럼 (경주 결과 후 확정되는 값들)
LEAKAGE_COLS = [
    "rsutRk",
    "rsutRaceRcd",
    "rsutMargin",
    "rsutRkAdmny",
    "rsutRkPurse",
    "rsutQnlaPrice",
    "rsutWinPrice",
    "rsutRkRemk",
    "rsutRlStrtTim",
    "rsutStrtTimChgRs",
]

# 타깃 컬럼
TARGET_COLS = ["target_rank", "target_is_top3"]

# 식별자 컬럼 (피처로 사용하지 않음)
ID_COLS = [
    "race_id", "schdRaceDt", "schdRaceNo",
    "pthrHrno", "pthrHrnm",
    "hrmJckyId", "hrmJckyNm",
    "hrmTrarId", "hrmTrarNm",
    "hrmOwnerId", "hrmOwnerNm",
]

# 최소 표본 수 기준 (이 미만이면 "참고용"으로 표시)
MIN_SAMPLE_N = 100

# Time-based split 기준일
TRAIN_CUTOFF = "2025-07-01"


# ─────────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────────

def setup_plot():
    """matplotlib 한글 폰트 + 스타일 설정 후 plt 반환"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    font_candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/NanumGothic.ttf",
        "C:/Windows/Fonts/gulim.ttc",
    ]
    for fc in font_candidates:
        if os.path.exists(fc):
            fe = fm.FontEntry(fname=fc, name="KoreanFont")
            fm.fontManager.ttflist.insert(0, fe)
            plt.rcParams["font.family"] = "KoreanFont"
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    return plt


def log(msg: str):
    """표준 출력 로그"""
    print(f"[INFO] {msg}", flush=True)


def log_step(n: int, title: str):
    """스텝 구분선 출력"""
    print(f"\n{'='*55}", flush=True)
    print(f"  Step {n}: {title}", flush=True)
    print(f"{'='*55}", flush=True)
