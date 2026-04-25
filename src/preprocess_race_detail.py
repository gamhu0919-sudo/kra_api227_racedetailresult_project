# kra_api227_racedetailresult_project/src/preprocess_race_detail.py

import pandas as pd
from pathlib import Path

from config import DIR_DATA_INTERIM, DIR_DATA_PROCESSED, DIR_LOGS, ANALYSIS_PERIOD_LABEL
from utils import CustomLogger, ensure_dir

def run_preprocessing(df: pd.DataFrame):
    """전처리 파이프라인 (타입 변환, 코드 정규화, 파생 변수 생성)"""
    logger = CustomLogger("preprocess_race_detail", DIR_LOGS).get_logger()
    
    if df.empty:
        logger.error("데이터셋이 비어 있습니다.")
        return
    
    logger.info(f"--- 전처리 시작 (기간: {ANALYSIS_PERIOD_LABEL}) ---")
    
    # 출력 폴더 보장
    ensure_dir(DIR_DATA_PROCESSED)
    
    # 1. 기초 복사 및 문자열 강제 변환 (안정적인 접근자 사용을 위해)
    processed_df = df.astype(str).copy()
    
    # 2. 문자열 기반 정제 (숫자 변환 전 수행)
    
    # 2-1. 나이(age) 숫자 추출 (예: "3세" -> 3)
    if 'age' in processed_df.columns:
        processed_df['age'] = processed_df['age'].str.extract(r'(\d+)')[0]

    # 2-2. 기록(rcTime) 초 단위 변환 (예: "1:12.3" -> 72.3)
    def time_to_seconds(val):
        if not isinstance(val, str) or ':' not in val: return None
        try:
            parts = val.split(':')
            return float(parts[0]) * 60 + float(parts[1])
        except: return None
    
    if 'rcTime' in processed_df.columns:
        processed_df['rcTime'] = processed_df['rcTime'].apply(time_to_seconds)

    # 2-3. 마체중(wgHr)에서 현재중량 및 증감값 분리 (예: 480(+2))
    if 'wgHr' in processed_df.columns:
        import re
        def extract_weight(val):
            if not isinstance(val, str): return None, None
            # 숫자(증감) 패턴 매칭
            match = re.search(r'(\d+)\(([\+\-]?[0-9]+)\)', val)
            if match:
                return float(match.group(1)), float(match.group(2))
            # 괄호 없는 단순 숫자만 있는 경우 처리
            match_simple = re.search(r'^(\d+)$', val)
            if match_simple:
                return float(match_simple.group(1)), 0.0
            return None, None
            
        weight_data = processed_df['wgHr'].apply(lambda x: pd.Series(extract_weight(x)))
        processed_df[['horse_weight', 'horse_weight_delta']] = weight_data

    # 3. 타입 변환 (정제된 필드 포함 숫자형으로 일괄 변환)
    numeric_cols = [
        'rcNo', 'stOrd', 'age', 'wgBudam', 'differ', 'win', 'plc', 'rcTime', 'hrRating',
        'horse_weight', 'horse_weight_delta'
    ]
    
    for col in numeric_cols:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # 4. 날짜형 변환
    if 'rcDate' in processed_df.columns:
        processed_df['rcDate_dt'] = pd.to_datetime(processed_df['rcDate'], format='%Y%m%d', errors='coerce')
    
    # 5. meet 코드 표준화 (1->서울, 2->제주, 3->부경)
    # 원본 meet 값이 문자열일 수 있으므로 유연하게 처리
    meet_map = {'1': "서울", '2': "제주", '3': "부경", 1: "서울", 2: "제주", 3: "부경"}
    if 'meet' in processed_df.columns:
        processed_df['meet_nm'] = processed_df['meet'].map(meet_map)
    
    # 6. 입상 여부 파생 변수
    processed_df['is_winner'] = (processed_df['stOrd'] == 1)
    processed_df['is_place'] = (processed_df['stOrd'] <= 3)
    
    # 7. 기간 필터 최종 적용 (2025.03~2026.03)
    processed_df = processed_df[processed_df['rcDate'].between("20250301", "20260331")].copy()
    
    # 8. 저장
    output_path = DIR_DATA_PROCESSED / "race_detail_result_202503_202603_processed.csv"
    processed_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"--- 전처리 완료. 저장 경로: {output_path} (행 수: {len(processed_df)}) ---")

if __name__ == "__main__":
    # 임시 테스트용
    interim_file = DIR_DATA_INTERIM / "race_detail_result_interim.csv"
    if interim_file.exists():
        df_test = pd.read_csv(interim_file)
        run_preprocessing(df_test)
