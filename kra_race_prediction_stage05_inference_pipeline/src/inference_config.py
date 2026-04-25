import os
import sys

# 1. 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
STAGE03_ROOT = os.path.join(ROOT_DIR, "kra_race_prediction_stage03_top3_modeling")
STAGE03_SRC = os.path.join(STAGE03_ROOT, "src")

# 데이터 경로
DATA_INPUT = os.path.join(BASE_DIR, "data", "input")
DATA_TEMPLATE = os.path.join(BASE_DIR, "data", "template")
DATA_REF = os.path.join(BASE_DIR, "data", "reference")
DATA_OUTPUT = os.path.join(BASE_DIR, "data", "output")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# 파일 경로
PATH_INPUT_ENTRIES = os.path.join(DATA_INPUT, "next_race_entries.csv")
PATH_TEMPLATE = os.path.join(DATA_TEMPLATE, "next_race_template.csv")
PATH_REF_HORSE = os.path.join(DATA_REF, "horse_latest_stats.csv")
PATH_REF_JOCKEY = os.path.join(DATA_REF, "jockey_latest_stats.csv")
PATH_REF_TRAINER = os.path.join(DATA_REF, "trainer_latest_stats.csv")

PATH_FE_BASE = os.path.join(DATA_OUTPUT, "inference_features_base.csv")
PATH_FE_RELATIVE = os.path.join(DATA_OUTPUT, "inference_features_relative.csv")
PATH_PREDICTIONS = os.path.join(DATA_OUTPUT, "next_race_predictions.csv")

# 모델 경로
PATH_MODEL = os.path.join(STAGE03_ROOT, "models", "lightgbm_top3_baseline.pkl")

# 공통 함수
def log(msg):
    print(f"[Stage05] {msg}")

def ensure_dirs():
    for d in [DATA_INPUT, DATA_TEMPLATE, DATA_REF, DATA_OUTPUT, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)
