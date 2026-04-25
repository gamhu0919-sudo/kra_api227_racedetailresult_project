"""
run_pipeline.py
──────────────────────────────
Stage 03: Top3 Baseline Modeling 전체 파이프라인 직렬 동기 실행
"""

import subprocess
import sys
import os
import time

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
PYTHON  = sys.executable

STEPS = [
    ("01_load_and_validate.py", "데이터 로드 및 Validation"),
    ("02_prepare_features.py", "결측 및 피처 처리"),
    ("03_time_split.py", "시간 기반 Split"),
    ("04_train_baseline_rules.py", "Baseline 규칙 예측 산출"),
    ("05_train_lightgbm_top3.py", "LightGBM 학습 및 확률 산출"),
    ("06_evaluate_race_level.py", "Race 단위 Metric 연산"),
    ("07_error_analysis.py", "오차 분석 (Test set 기준)"),
    ("08_make_final_modeling_report.py", "결과 통합 Markdown 렌더링"),
]

print("="*60)
print("KRA Stage 03 - Top3 Baseline Modeling 파이프라인 개시")
print("="*60)

total_start = time.time()
failures = []

for script, desc in STEPS:
    path = os.path.join(SRC_DIR, script)
    print(f"\n[실행] {script} - {desc}")
    start_time = time.time()
    
    # stdout을 바로 출력하도록 설정
    result = subprocess.run([PYTHON, path], capture_output=False, text=True)
    
    elapsed = round(time.time() - start_time, 1)
    if result.returncode == 0:
        print(f"[OK] {script} 완료 ({elapsed}초)")
    else:
        print(f"[실패] {script} 실행 중 오류가 발생했습니다. (Return Code: {result.returncode})")
        failures.append(script)
        break # 중대 오류 시 즉시 중단

total_elapsed = round(time.time() - total_start, 1)

print("\n" + "="*60)
if not failures:
    print(f"전체 파이프라인 완수! 총 소요 라우팅 시간: {total_elapsed}초")
else:
    print(f"파이프라인 중단. 실패 파일명: {failures}")
print("="*60)
