import os
import pandas as pd

# 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STAGE03_PROCESSED_DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "kra_race_prediction_stage03_top3_modeling", "data", "processed", "modeling_data_validated.csv"))

REF_DIR = os.path.join(BASE_DIR, "data", "reference")
os.makedirs(REF_DIR, exist_ok=True)

def main():
    print(f"Reading historical validated data from:\n{STAGE03_PROCESSED_DATA_PATH}")
    # hrmJckyId, hrmTrarId 등은 0으로 시작하는 문자열이므로 dtype 지정
    df = pd.read_csv(STAGE03_PROCESSED_DATA_PATH, dtype={'hrmJckyId': str, 'hrmTrarId': str}, low_memory=False)
    
    # 시간 순 정렬을 위해 날짜 캐싱 또는 정렬
    if 'schdRaceDt' in df.columns:
        df['schdRaceDt'] = pd.to_datetime(df['schdRaceDt'])
        df = df.sort_values('schdRaceDt')
    
    # 1. 말 통계 (가장 마지막 누적 승률, 누적 평균순위, 출전수, 마지막 체중)
    # 추론 시 당일 마체중(fe_horse_weight)을 직접 입력하면, (입력 체중 - 마지막 체중)으로 증감연산 처리
    print("Extracting horse latest stats...")
    horse_cols = ['pthrHrno', 'fe_horse_cum_win_rate', 'fe_horse_cum_avg_rk', 'fe_horse_race_count', 'fe_horse_weight']
    horse_stats = df.drop_duplicates(subset=['pthrHrno'], keep='last')[horse_cols]
    horse_stats = horse_stats.rename(columns={'fe_horse_weight': 'last_horse_weight'})
    horse_stats.to_csv(os.path.join(REF_DIR, "horse_latest_stats.csv"), index=False, encoding="utf-8-sig")
    
    # 2. 기수 통계
    print("Extracting jockey latest stats...")
    jcky_cols = ['hrmJckyId', 'fe_jcky_cum_win_rate', 'fe_jcky_cum_top3_rate']
    jcky_stats = df.drop_duplicates(subset=['hrmJckyId'], keep='last')[jcky_cols]
    jcky_stats.to_csv(os.path.join(REF_DIR, "jockey_latest_stats.csv"), index=False, encoding="utf-8-sig")
    
    # 3. 조교사 통계
    print("Extracting trainer latest stats...")
    trar_cols = ['hrmTrarId', 'fe_trar_cum_win_rate']
    trar_stats = df.drop_duplicates(subset=['hrmTrarId'], keep='last')[trar_cols]
    trar_stats.to_csv(os.path.join(REF_DIR, "trainer_latest_stats.csv"), index=False, encoding="utf-8-sig")
    
    print("Reference data generation completed.")

if __name__ == "__main__":
    main()
