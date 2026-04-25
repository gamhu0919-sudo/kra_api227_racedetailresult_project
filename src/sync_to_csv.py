# kra_api227_racedetailresult_project/src/sync_to_csv.py

import os
import pandas as pd
from pathlib import Path

from config import DIR_DATA_RAW_XML, DIR_DATA_INTERIM, DIR_LOGS, ANALYSIS_PERIOD_LABEL
from utils import CustomLogger, ensure_dir
from xml_parser import KRA_XMLParser
from preprocess_race_detail import run_preprocessing

def run_sync():
    """수집된 모든 XML을 파싱하여 중간 CSV로 저장하고 전처리를 호출함"""
    logger = CustomLogger("sync_to_csv", DIR_LOGS).get_logger()
    parser = KRA_XMLParser(logger)
    
    logger.info(f"--- XML to CSV 동기화 시작 (대상: {DIR_DATA_RAW_XML}) ---")
    
    # 1. XML 파일 목록 확보
    xml_files = list(DIR_DATA_RAW_XML.glob("*.xml"))
    logger.info(f"  발견된 XML 파일 수: {len(xml_files)}개")
    
    if not xml_files:
        logger.warning("  파싱할 XML 파일이 없습니다.")
        return
        
    # 2. 파싱 및 병합
    logger.info("  XML 파싱 및 데이터 병합 중...")
    df_merged = parser.merge_raw_files_to_interim([str(f) for f in xml_files])
    
    if df_merged.empty:
        logger.warning("  유효한 데이터가 추출되지 않았습니다.")
        return
        
    logger.info(f"  병합 완료: {len(df_merged)}행 추출")
    
    # 3. 중간 데이터 저장 (Interim)
    ensure_dir(DIR_DATA_INTERIM)
    interim_path = DIR_DATA_INTERIM / "race_detail_result_interim.csv"
    df_merged.to_csv(interim_path, index=False, encoding='utf-8-sig')
    logger.info(f"  중간 CSV 저장 완료: {interim_path}")
    
    # 4. 전처리 파이프라인 즉시 호출 (Processed CSV 생성)
    logger.info("  전처리 파이프라인(preprocess_race_detail) 호출...")
    run_preprocessing(df_merged)
    
    logger.info(f"--- 동기화 완료! 최종 CSV가 data/processed 폴더에 생성되었습니다. ---")

if __name__ == "__main__":
    run_sync()
