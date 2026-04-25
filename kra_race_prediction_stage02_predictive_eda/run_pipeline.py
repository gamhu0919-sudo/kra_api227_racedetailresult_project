"""
run_pipeline.py
─────────────────────────────────
Stage 02 예측형 EDA 전체 파이프라인 실행

실행 순서:
  01_data_validation.py
  02_predictive_eda.py
  03_feature_signal_check.py
  04_leakage_check.py
  05_make_modeling_dataset.py
  06_generate_final_report.py
"""

import subprocess
import sys
import os
import time

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
PYTHON  = sys.executable

STEPS = [
    ("01_data_validation.py",       "데이터 로드 및 정합성 검증"),
    ("02_predictive_eda.py",         "변수별 Lift 분석"),
    ("03_feature_signal_check.py",   "경주 내 상대 피처 생성 + 신호 검증"),
    ("04_leakage_check.py",          "누적 피처 누수 검증"),
    ("05_make_modeling_dataset.py",  "베이스라인 + 모델링 데이터셋 생성"),
    ("06_generate_final_report.py",  "최종 보고서 생성"),
]

print("="*60)
print("KRA Stage 02 예측형 EDA 파이프라인 시작")
print("="*60)

total_start = time.time()
failures = []

for script, desc in STEPS:
    path = os.path.join(SRC_DIR, script)
    print(f"\n[실행] {script} — {desc}")
    start = time.time()
    result = subprocess.run(
        [PYTHON, path],
        capture_output=False,
        text=True,
    )
    elapsed = round(time.time() - start, 1)
    if result.returncode == 0:
        print(f"[OK] {script} 완료 ({elapsed}초)")
    else:
        print(f"[실패] {script} 오류 발생 (종료코드: {result.returncode})")
        failures.append(script)

total_elapsed = round(time.time() - total_start, 1)

print("\n" + "="*60)
if not failures:
    print(f"전체 파이프라인 완료 ({total_elapsed}초)")
else:
    print(f"완료 ({total_elapsed}초) — 실패 스크립트: {failures}")
print("="*60)
