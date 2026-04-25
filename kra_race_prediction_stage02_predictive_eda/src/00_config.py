"""
공통 경로 및 설정 모듈
모든 스크립트에서 import하여 사용
"""

import os
import sys
import warnings

# Windows 콘솔 UTF-8 출력 강제
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 경로 정의
# ─────────────────────────────────────────────
# 이 파일의 위치: .../kra_race_prediction_stage02_predictive_eda/src/
STAGE02_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 상위 프로젝트 루트 (kra_api227_racedetailresult_project)
PROJECT_ROOT = os.path.dirname(STAGE02_DIR)

# 원본 데이터 경로
RAW_CSV  = os.path.join(PROJECT_ROOT, "kra_race_prediction_eda", "race_results_seoul_3years_revised.csv")
SCHEMA_XL = os.path.join(PROJECT_ROOT, "kra_race_prediction_eda", "schema_info.xlsx")

# Stage02 내부 경로
DATA_RAW       = os.path.join(STAGE02_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(STAGE02_DIR, "data", "processed")
DATA_MODELING  = os.path.join(STAGE02_DIR, "data", "modeling_ready")
OUT_TABLES     = os.path.join(STAGE02_DIR, "outputs", "tables")
OUT_FIGS       = os.path.join(STAGE02_DIR, "outputs", "figures")
OUT_DIAG       = os.path.join(STAGE02_DIR, "outputs", "diagnostics")
REPORTS        = os.path.join(STAGE02_DIR, "reports")

for p in [DATA_RAW, DATA_PROCESSED, DATA_MODELING,
          OUT_TABLES, OUT_FIGS, OUT_DIAG, REPORTS]:
    os.makedirs(p, exist_ok=True)

# ─────────────────────────────────────────────
# 누수 컬럼 정의
# ─────────────────────────────────────────────
LEAKAGE_COLS = [
    "rsutRk",            # 타깃 원본 (직접 누수)
    "rsutRaceRcd",       # 레이스 타임 (경주 후)
    "rsutMargin",        # 마신차 (경주 후)
    "rsutRkAdmny",       # 입상 여부 (경주 후)
    "rsutRkPurse",       # 상금 (경주 후)
    "rsutQnlaPrice",     # 연세가격 (경주 후)
    "rsutWinPrice",      # 배당금 (경주 후 확정)
    "rsutRkRemk",        # 비고 (경주 후)
    "rsutRlStrtTim",     # 실제 출발 시각 (경주 후)
    "rsutStrtTimChgRs",  # 출발 시각 변경 사유 (경주 후)
]

# 타깃 컬럼 (모델 입력에서 제외하되 타깃으로 사용)
TARGET_COLS = ["target_rank", "target_is_top3"]

# 식별자 컬럼 (피처에서 제외)
ID_COLS = ["race_id", "schdRaceDt", "schdRaceNo", "pthrHrno", "pthrHrnm",
           "hrmJckyId", "hrmJckyNm", "hrmTrarId", "hrmTrarNm",
           "hrmOwnerId", "hrmOwnerNm"]

# 최소 표본 수 기준
MIN_SAMPLE_N = 100

# ─────────────────────────────────────────────
# 한글 폰트 설정 (matplotlib)
# ─────────────────────────────────────────────
def setup_plot():
    """matplotlib 한글 폰트 + 스타일 설정"""
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

def log(msg):
    print(f"[INFO] {msg}", flush=True)

def log_step(n, title):
    print(f"\n{'='*55}", flush=True)
    print(f"  Step {n}: {title}", flush=True)
    print(f"{'='*55}", flush=True)
