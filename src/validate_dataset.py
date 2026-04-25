# kra_api227_racedetailresult_project/src/validate_dataset.py

import pandas as pd
import os
from pathlib import Path

from config import DIR_DATA_INTERIM, DIR_LOGS, ANALYSIS_PERIOD_LABEL
from utils import CustomLogger

def run_validation(df: pd.DataFrame):
    """데이터셋 무결성 검증"""
    logger = CustomLogger("validate_dataset", DIR_LOGS).get_logger()
    
    if df.empty:
        logger.error("데이터셋이 비어 있습니다. 수집을 먼저 진행하세요.")
        return
    
    logger.info(f"--- 데이터셋 검증 시작 (기간: {ANALYSIS_PERIOD_LABEL}) ---")
    
    # 1. 상하위 행 출력
    logger.info(f"Head(5):\n{df.head(5)}")
    logger.info(f"Tail(5):\n{df.tail(5)}")
    
    # 2. DataFrame 정보 (info)
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    logger.info(f"DataFrame Info:\n{buffer.getvalue()}")
    
    # 3. 전체 행/열 수
    logger.info(f"Shape: {df.shape}")
    
    # 4. 결측치 비율
    null_counts = df.isnull().sum()
    null_ratio = (null_counts / len(df)) * 100
    missing_report = pd.DataFrame({"null_count": null_counts, "ratio_pct": null_ratio})
    logger.info(f"Missing Value Report (NaN/Null):\n{missing_report[missing_report['null_count'] > 0]}")
    
    # 5. 중복 데이터 (기본키 candidate: meet+rcDate+rcNo+chulNo)
    # API227은 경주별 상세 성적표이므로 (경주+출전마)가 유니크해야 함
    if 'meet' in df.columns and 'rcDate' in df.columns and 'rcNo' in df.columns and 'chulNo' in df.columns:
        dup_cols = ['meet', 'rcDate', 'rcNo', 'chulNo']
        duplicates = df.duplicated(subset=dup_cols).sum()
        logger.info(f"Duplicate Row Count (by {dup_cols}): {duplicates}")
    
    # 6. 기간 범위 검증
    if 'rcDate' in df.columns:
        min_date = df['rcDate'].min()
        max_date = df['rcDate'].max()
        logger.info(f"Date Range in Dataset: {min_date} ~ {max_date}")
        
        # 2025.03 ~ 2026.03 범위 밖 데이터 존재 여부
        out_of_range = df[~df['rcDate'].between("20250301", "20260331")].shape[0]
        if out_of_range > 0:
            logger.warning(f"  [Warning] 기간 범위 밖 데이터 {out_of_range}건 발견!")
    
    # 7. 컬럼별 고유값 수
    nunique = df.nunique()
    logger.info(f"Unique Values Count per Column:\n{nunique}")
    
    # 8. 이상치 후보 탐지 (하이픈, 특정 센티넬 값 등)
    sentinel_candidates = ["-", "9999.9", "0.0", ""]
    for val in sentinel_candidates:
        counts = (df == val).sum().sum()
        if counts > 0:
            logger.info(f"  Found potential sentinel value '{val}': {counts} instances")

    logger.info("--- 검증 완료 ---")

if __name__ == "__main__":
    # 임시 테스트용 (파일 있으면 실행)
    interim_file = DIR_DATA_INTERIM / "race_detail_result_interim.csv"
    if interim_file.exists():
        df_test = pd.read_csv(interim_file)
        run_validation(df_test)
