import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STAGE03_PROCESSED_DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "kra_race_prediction_stage03_top3_modeling", "data", "processed", "modeling_data_validated.csv"))
TEMPLATE_DIR = os.path.join(BASE_DIR, "data", "template")
os.makedirs(TEMPLATE_DIR, exist_ok=True)

def main():
    print(f"Reading historical validated data to extract a sample next race...")
    df = pd.read_csv(STAGE03_PROCESSED_DATA_PATH, dtype={'hrmJckyId': str, 'hrmTrarId': str}, low_memory=False)
    
    if 'schdRaceDt' in df.columns:
        df['schdRaceDt'] = pd.to_datetime(df['schdRaceDt'])
        df = df.sort_values('schdRaceDt')
    
    # 마지막 경주 하나를 선택 
    last_race_id = df['race_id'].iloc[-1]
    sample_race = df[df['race_id'] == last_race_id].copy()
    
    # Inference에 필요한 컬럼 정의 (사용자 입력 대상)
    template_cols = [
        'race_id', 'schdRaceDt', 'schdRaceNo', 
        'pthrHrno', 'pthrHrnm', 'hrmJckyId', 'hrmJckyNm', 'hrmTrarId', 'hrmTrarNm', 
        'pthrBurdWgt', 'pthrGtno', 'pthrRatg', 'cndRaceClas', 'cndBurdGb', 'cndGndr', 'cndAg', 'cndRatg', 'fe_race_dist', 
        'hrmJckyAlw', 'fe_horse_weight', 'rsutWetr', 'rsutTrckStus', 'fe_track_humidity'
    ]
    
    # 누락된 컬럼에 대해 일단 채우기
    present_cols = [c for c in template_cols if c in sample_race.columns]
    
    # 저장
    sample_template = sample_race[present_cols].copy()
    
    # 날짜 포맷 되돌리기
    sample_template['schdRaceDt'] = sample_template['schdRaceDt'].dt.strftime('%Y-%m-%d')
    sample_template.to_csv(os.path.join(TEMPLATE_DIR, "next_race_template.csv"), index=False, encoding="utf-8-sig")
    print(f"Sample template created at {TEMPLATE_DIR}/next_race_template.csv")

if __name__ == "__main__":
    main()
