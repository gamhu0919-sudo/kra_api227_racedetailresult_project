# kra_api227_racedetailresult_project/src/eda_race_detail.py

import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from pathlib import Path

from config import DIR_DATA_PROCESSED, DIR_IMAGES, DIR_LOGS, ANALYSIS_PERIOD_LABEL
from utils import CustomLogger

def run_eda(df: pd.DataFrame):
    """기초 EDA 및 시각화 (matplotlib 기반)"""
    logger = CustomLogger("eda_race_detail", DIR_LOGS).get_logger()
    
    if df.empty:
        logger.error("데이터셋이 비어 있습니다.")
        return
    
    logger.info(f"--- 기초 EDA 시작 (기간: {ANALYSIS_PERIOD_LABEL}) ---")
    
    # 1. 수치형 변수 기술통계
    desc = df.describe()
    logger.info(f"Descriptive Statistics:\n{desc}")
    
    # 2. 범주형 변수 빈도 분석 (Top 30)
    # 기수명(jkName), 말명(hrName), 조교사명(trName) 등
    cat_cols = ['jkName', 'hrName', 'trName', 'owName', 'meet_nm', 'sex']
    
    for col in cat_cols:
        if col in df.columns:
            counts = df[col].value_counts().head(30)
            
            # 빈도표 출력
            logger.info(f"Top 30 Frequencies for '{col}':\n{counts}")
            
            # 시각화 (막대그래프)
            plt.figure(figsize=(12, 6))
            counts.plot(kind='bar', color='skyblue', edgecolor='black')
            plt.title(f"{col} 빈도수 (Top 30) - {ANALYSIS_PERIOD_LABEL}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            img_path = DIR_IMAGES / f"bar_freq_{col}.png"
            plt.savefig(img_path)
            plt.close()
            logger.info(f"  Saved plot: {img_path}")

    # 3. 입상마(1위) 기준 핵심 집계
    if 'stOrd' in df.columns:
        winners = df[df['stOrd'] == 1]
        
        # 1위 최다 기수
        top_jockeys = winners['jkName'].value_counts().head(10)
        logger.info(f"Top 10 Winning Jockeys:\n{top_jockeys}")
        
        # 1위 최다 마필
        top_horses = winners['hrName'].value_counts().head(10)
        logger.info(f"Top 10 Winning Horses:\n{top_horses}")

    # 4. 경마장별 성적 비율
    if 'meet_nm' in df.columns:
        meet_counts = df['meet_nm'].value_counts()
        plt.figure(figsize=(8, 8))
        meet_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['gold', 'lightcoral', 'lightskyblue'])
        plt.title(f"경마장별 출전 비중 - {ANALYSIS_PERIOD_LABEL}")
        plt.ylabel("")
        plt.tight_layout()
        
        img_path = DIR_IMAGES / "pie_meet_distribution.png"
        plt.savefig(img_path)
        plt.close()

    logger.info("--- 기초 EDA 완료 ---")

if __name__ == "__main__":
    # 임시 테스트용
    processed_file = DIR_DATA_PROCESSED / "race_detail_result_202503_202603_processed.csv"
    if processed_file.exists():
        df_test = pd.read_csv(processed_file)
        run_eda(df_test)
