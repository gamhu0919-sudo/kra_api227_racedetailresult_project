import os
import pandas as pd
import numpy as np
import re

def convert_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def main():
    print("=== API15_2 Interim CSV 전처리(Preprocessing) 시작 ===")
    
    interim_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'interim', 'api15_2', 'parsed_api15_2.csv')
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'api15_2')
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, 'processed_api15_2.csv')
    
    if not os.path.exists(interim_path):
        print(f"{interim_path} 가 존재하지 않습니다. 앞선 단계를 선행하세요.")
        return
        
    df = pd.read_csv(interim_path, encoding='utf-8-sig')
    
    # 1. 컬럼명 일원화 (CamelCase -> snake_case)
    df.columns = [convert_snake_case(col) for col in df.columns]
    
    # 핵심 식별자 소실(마번 등)시 Drop
    if 'hr_no' in df.columns:
        initial_len = len(df)
        df.dropna(subset=['hr_no'], inplace=True)
        if len(df) < initial_len:
            print(f"    - HR_NO 결측치 완전 삭제: {initial_len - len(df)} rows")
    else:
        print("    [!] 경고: hr_no 식별자 컬럼이 존재하지 않습니다.")

    # 2. 날짜/수치 변환 및 결측치 치환 규칙 적용
    # 날짜형 (가이드: debutDt, recentRcDate 등)
    date_cols = ['debut_dt', 'recent_rc_date', 'debut_date'] # 원본 파생에 따른 복수에러 방어
    for col in date_cols:
        if col in df.columns:
            # ex: '20250325' 등을 실제 시한으로 캐스팅
            df[col] = pd.to_datetime(df[col].astype(str).str.replace(r'\D', '', regex=True), format='%Y%m%d', errors='coerce')
    
    # 수치형 (통산 상금, 승률, 출주횟수 등 모두 0으로 FillNA)
    numeric_keywords = ['cnt', 'rate', 'prize', 'wt', 'time', 'rating', 'age', 'no']
    for col in df.columns:
        if any(k in col for k in numeric_keywords) and 'hr_no' not in col:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # 그외 텍스트 범주의 결측치
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        df[col] = df[col].fillna("Unknown")
        
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Processed 데이터 저장 완료: {out_path} (Shape: {df.shape})")

if __name__ == "__main__":
    main()
