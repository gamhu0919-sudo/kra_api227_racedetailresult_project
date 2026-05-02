import os
import pandas as pd
import inference_config as cfg

def main():
    cfg.ensure_dirs()
    cfg.log("기준 테이블(Reference Tables) 생성을 시작합니다.")
    
    # 1. 과거 검증 데이터 로드 (v2 데이터셋 사용)
    hist_path = os.path.join(cfg.STAGE03_V2_ROOT, "data", "processed", "modeling_data_v2_final.csv")
    if not os.path.exists(hist_path):
        # 만약 final이 없으면 processed의 마지막 단계라도 사용
        hist_path = os.path.join(cfg.STAGE03_V2_ROOT, "data", "processed", "modeling_data_dist_class.csv")
        
    if not os.path.exists(hist_path):
        cfg.log(f"Error: 원천 데이터 누락 ({hist_path})")
        return

    # ID 컬럼 문자열 보존을 위해 dtype 지정
    df = pd.read_csv(hist_path, dtype={'hrmJckyId': str, 'hrmTrarId': str, 'pthrHrno': str}, low_memory=False)
    
    # 날짜순 정렬
    if 'schdRaceDt' in df.columns:
        df['schdRaceDt'] = pd.to_datetime(df['schdRaceDt'])
        df = df.sort_values(['schdRaceDt', 'race_id'])

    # 1. 말 통계 (v2 신규 피처 대거 포함)
    cfg.log("말 기준 테이블 생성 중...")
    # 신규 피처 리스트 자동 추출 (fe_horse_로 시작하거나 fe_로 시작하는 말 관련)
    horse_fe_cols = [c for c in df.columns if c.startswith('fe_horse_')]
    horse_cols = ['pthrHrno', 'fe_horse_weight'] + horse_fe_cols
    # 중복 제거 (pthrHrno 기준)
    horse_cols = list(set(horse_cols)) 
    
    # 마지막 경주 시점의 누적 스탯을 추론 시점의 '현재 스탯'으로 사용
    horse_stats = df.drop_duplicates(subset=['pthrHrno'], keep='last')[horse_cols].copy()
    horse_stats = horse_stats.rename(columns={'fe_horse_weight': 'last_horse_weight'})
    
    # 휴식일수 계산을 위해 마지막 경주일자 저장
    if 'schdRaceDt' in df.columns:
        horse_stats['last_race_date'] = df.drop_duplicates(subset=['pthrHrno'], keep='last')['schdRaceDt'].values

    horse_stats.to_csv(cfg.PATH_REF_HORSE, index=False, encoding="utf-8-sig")

    # 2. 기수 통계
    cfg.log("기수 기준 테이블 생성 중...")
    jcky_fe_cols = [c for c in df.columns if c.startswith('fe_jcky_')]
    jcky_cols = ['hrmJckyId'] + jcky_fe_cols
    jcky_stats = df.drop_duplicates(subset=['hrmJckyId'], keep='last')[jcky_cols].copy()
    jcky_stats.to_csv(cfg.PATH_REF_JOCKEY, index=False, encoding="utf-8-sig")

    # 3. 조교사 통계
    cfg.log("조교사 기준 테이블 생성 중...")
    trar_fe_cols = [c for c in df.columns if c.startswith('fe_trar_')]
    trar_cols = ['hrmTrarId'] + trar_fe_cols
    trar_stats = df.drop_duplicates(subset=['hrmTrarId'], keep='last')[trar_cols].copy()
    trar_stats.to_csv(cfg.PATH_REF_TRAINER, index=False, encoding="utf-8-sig")

    cfg.log(f"파일 저장 완료: {cfg.DATA_REF} 폴더 내 3개 파일 (v2 피처 반영)")

if __name__ == "__main__":
    main()
