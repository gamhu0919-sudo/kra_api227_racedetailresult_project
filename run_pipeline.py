# -*- coding: utf-8 -*-
"""
run_pipeline.py - 전체 파이프라인 순차 실행
실행: python run_pipeline.py

순서:
  1. data_audit.py   - 데이터 감사
  2. preprocess.py   - 전처리
  3. feature_engineering.py - 피처 엔지니어링
  4. train_baselines.py     - 베이스라인
  5. train_ranker.py        - Ranker 모델
  6. train_classifiers.py   - 분류기
  7. calibration.py         - 확률 보정
  8. evaluate.py            - 통합 평가
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils import get_logger
from config import LOGS_DIR

logger = get_logger("run_pipeline", LOGS_DIR)


def run_step(step_name: str, func):
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {step_name}")
    logger.info(f"{'='*60}")
    t0 = time.time()
    try:
        result = func()
        elapsed = time.time() - t0
        logger.info(f"  ✓ {step_name} 완료 ({elapsed:.1f}초)")
        return result
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"  ✗ {step_name} 실패 ({elapsed:.1f}초): {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    total_start = time.time()
    logger.info("전체 파이프라인 시작")

    # Step 1: 데이터 감사
    import data_audit
    run_step("1. 데이터 감사", data_audit.main)

    # Step 2: 전처리
    import preprocess
    run_step("2. 전처리", preprocess.main)

    # Step 3: 피처 엔지니어링
    import feature_engineering
    run_step("3. 피처 엔지니어링", feature_engineering.main)

    # Step 4: 베이스라인
    import train_baselines
    run_step("4. 베이스라인", train_baselines.main)

    # Step 5: Ranker
    import train_ranker
    run_step("5. Ranker 학습", train_ranker.main)

    # Step 6: 분류기
    import train_classifiers
    run_step("6. 분류기 학습", train_classifiers.main)

    # Step 7: 보정
    import calibration
    run_step("7. 확률 보정", calibration.main)

    # Step 8: 평가
    import evaluate
    run_step("8. 통합 평가", evaluate.main)

    total = time.time() - total_start
    logger.info(f"\n전체 파이프라인 완료 (총 {total/60:.1f}분)")
    logger.info("Streamlit 앱 실행: streamlit run streamlit_app.py")
