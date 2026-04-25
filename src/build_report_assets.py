# kra_api227_racedetailresult_project/src/build_report_assets.py

import os
import shutil
from pathlib import Path
import pandas as pd

from config import DIR_DATA_PROCESSED, DIR_IMAGES, DIR_DOCS, ANALYSIS_PERIOD_LABEL

def run_asset_cleaning():
    """보고서용 최종 자산 정리 및 요약"""
    print(f"--- 리포트 자산 정리 시작 (기간: {ANALYSIS_PERIOD_LABEL}) ---")
    
    # 1. 최종 가공 데이터 확인
    processed_file = DIR_DATA_PROCESSED / "race_detail_result_202503_202603_processed.csv"
    if processed_file.exists():
        df = pd.read_csv(processed_file)
        print(f"  Final Dataset: {processed_file} (Rows: {len(df)})")
        
        # 간단한 요약표 생성
        summary_df = df.groupby('meet_nm')['is_winner'].sum().reset_index()
        summary_df.columns = ['경마장', '우승마건수']
        summary_path = DIR_DATA_PROCESSED / "summary_by_meet.csv"
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"  Summary saved: {summary_path}")

    # 2. 이미지 파일 확인
    images = list(DIR_IMAGES.glob("*.png"))
    print(f"  Generated Visualization Files ({len(images)}):")
    for img in images:
        print(f"    - {img.name}")

    # 3. 작업 로그 기록 보조 (md 파일 용)
    assets_report_path = DIR_DOCS / "generated_assets_list.md"
    with open(assets_report_path, "w", encoding='utf-8') as f:
        f.write(f"# 생성된 자산 목록 ({ANALYSIS_PERIOD_LABEL})\n\n")
        f.write("## 1. 정제 데이터셋\n")
        f.write(f"- [race_detail_result_processed.csv](file:///{processed_file})\n")
        f.write("## 2. 시각화 이미지\n")
        for img in images:
            f.write(f"- ![{img.name}](file:///{img})\n")
            
    print(f"--- 리포트 자산 정리 완료. 목록 확인: {assets_report_path} ---")

if __name__ == "__main__":
    run_asset_cleaning()
