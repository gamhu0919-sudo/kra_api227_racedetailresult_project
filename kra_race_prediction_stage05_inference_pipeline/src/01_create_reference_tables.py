import os
import pandas as pd
import inference_config as cfg

def main():
    cfg.ensure_dirs()
    cfg.log("기준 테이블(Reference Tables) 생성을 시작합니다.")
    
    # 1. 과거 검증 데이터 로드
    hist_path = os.path.join(cfg.STAGE03_ROOT, "data", "processed", "modeling_data_validated.csv")
    if not os.path.exists(hist_path):
        cfg.log(f"Error: 원천 데이터 누락 ({hist_path})")
        return

    # ID 컬럼 문자열 보존을 위해 dtype 지정
    df = pd.read_csv(hist_path, dtype={'hrmJckyId': str, 'hrmTrarId': str, 'pthrHrno': str}, low_memory=False)
    
    # 날짜순 정렬
    if 'schdRaceDt' in df.columns:
        df['schdRaceDt'] = pd.to_datetime(df['schdRaceDt'])
        df = df.sort_values(['schdRaceDt', 'race_id'])

    # 1. 말 통계 (가장 마지막 누적 승률, 평균순위, 출전수, 체중)
    cfg.log("말 기준 테이블 생성 중...")
    horse_cols = ['pthrHrno', 'fe_horse_cum_win_rate', 'fe_horse_cum_avg_rk', 'fe_horse_race_count', 'fe_horse_weight']
    horse_stats = df.drop_duplicates(subset=['pthrHrno'], keep='last')[horse_cols].copy()
    horse_stats = horse_stats.rename(columns={'fe_horse_weight': 'last_horse_weight'})
    horse_stats.to_csv(cfg.PATH_REF_HORSE, index=False, encoding="utf-8-sig")

    # 2. 기수 통계
    cfg.log("기수 기준 테이블 생성 중...")
    jcky_cols = ['hrmJckyId', 'fe_jcky_cum_win_rate', 'fe_jcky_cum_top3_rate']
    jcky_stats = df.drop_duplicates(subset=['hrmJckyId'], keep='last')[jcky_cols].copy()
    jcky_stats.to_csv(cfg.PATH_REF_JOCKEY, index=False, encoding="utf-8-sig")

    # 3. 조교사 통계
    cfg.log("조교사 기준 테이블 생성 중...")
    trar_cols = ['hrmTrarId', 'fe_trar_cum_win_rate']
    trar_stats = df.drop_duplicates(subset=['hrmTrarId'], keep='last')[trar_cols].copy()
    trar_stats.to_csv(cfg.PATH_REF_TRAINER, index=False, encoding="utf-8-sig")

    cfg.log(f"파일 저장 완료: {cfg.DATA_REF} 폴더 내 3개 파일")

if __name__ == "__main__":
    main()
