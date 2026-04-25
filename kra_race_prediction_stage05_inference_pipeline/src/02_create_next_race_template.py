import os
import pandas as pd
import inference_config as cfg

def main():
    cfg.ensure_dirs()
    cfg.log("입력 템플릿(Template) 생성을 시작합니다.")
    
    # 1. 과거 데이터에서 샘플 추출
    hist_path = os.path.join(cfg.STAGE03_ROOT, "data", "processed", "modeling_data_validated.csv")
    if not os.path.exists(hist_path):
        cfg.log("Error: 과거 데이터를 찾을 수 없어 템플릿을 생성할 수 없습니다.")
        return

    hist_df = pd.read_csv(hist_path, dtype={'hrmJckyId': str, 'hrmTrarId': str, 'pthrHrno': str}, low_memory=False)
    
    # 마지막 경주 ID 선택
    last_race_id = hist_df['race_id'].iloc[-1]
    sample_race = hist_df[hist_df['race_id'] == last_race_id].copy()

    # 필수 입력 컬럼 정의
    required_cols = [
        'race_id', 'schdRaceDt', 'schdRaceNo', 
        'pthrHrno', 'pthrHrnm', 'hrmJckyId', 'hrmJckyNm', 'hrmTrarId', 'hrmTrarNm', 
        'pthrBurdWgt', 'pthrGtno', 'pthrRatg', 'cndRaceClas', 'cndBurdGb', 'cndGndr', 
        'cndAg', 'cndRatg', 'fe_race_dist', 'hrmJckyAlw', 'fe_horse_weight', 
        'rsutWetr', 'rsutTrckStus', 'fe_track_humidity'
    ]

    # 샘플 데이터에 존재하는 컬럼만 추출
    available_cols = [c for c in required_cols if c in sample_race.columns]
    template_df = sample_race[available_cols].copy()

    # 템플릿 저장
    template_df.to_csv(cfg.PATH_TEMPLATE, index=False, encoding="utf-8-sig")
    cfg.log(f"템플릿 생성 완료: {cfg.PATH_TEMPLATE}")

    # 실제 입력 위치에도 샘플 복사 (자동 테스트 용)
    template_df.to_csv(cfg.PATH_INPUT_ENTRIES, index=False, encoding="utf-8-sig")
    cfg.log(f"샘플 입력 데이터 복사 완료: {cfg.PATH_INPUT_ENTRIES}")

if __name__ == "__main__":
    main()
