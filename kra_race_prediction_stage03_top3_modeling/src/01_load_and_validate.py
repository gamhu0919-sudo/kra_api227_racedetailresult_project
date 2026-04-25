"""
01_load_and_validate.py
──────────────────────────────
이전 단계의 모델링 데이터셋을 로드하고 누수 피처 검증, 기본 정보 로그
"""

import os
import sys
import pandas as pd

# 설정 모듈이 있는 상위 디렉터리를 path로 등재 (실행 위치 무관하게 동작하도록)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

config.log_step(1, "모델링 인풋 데이터 로드 및 누수 검증")

def main():
    if not os.path.exists(config.PREVQ_MODELING):
        config.log(f"[ERROR] 1순위 입력 파일이 존재하지 않습니다: {config.PREVQ_MODELING}")
        # Note: 사용자 지침에 2순위 데이터가 있지만, 
        # 이전 작업에서 미리 생성했으므로 예외 처리보단 존재를 강하게 Assert 합니다.
        raise FileNotFoundError(config.PREVQ_MODELING)
    
    config.log(f"파일 존재 확인 OK: {config.PREVQ_MODELING}")
    df = pd.read_csv(config.PREVQ_MODELING, encoding="utf-8-sig", low_memory=False)
    config.log(f"데이터 로드 완료. (Shape: {df.shape})")

    # 1. Leakage Check
    # target_is_top3는 라벨이므로 유지하고 나머지 List 차단.
    drop_targets = [c for c in config.LEAKAGE_COLS if c != config.TARGET_COL]
    
    leaks_found = []
    for c in drop_targets:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
            leaks_found.append(c)

    if leaks_found:
        config.log(f"누수 컬럼 발견되어 삭제 조치됨: {leaks_found}")
    else:
        config.log("누수 컬럼(Leakage) 모델 입력단 완벽 차단 확인.")

    # 2. 기초 저장 
    summary_data = {
        "Total Rows": len(df),
        "Total Races": df["race_id"].nunique(),
        "Date Min": df["schdRaceDt"].min(),
        "Date Max": df["schdRaceDt"].max(),
        "Columns Intact": len(df.columns),
    }

    summary_df = pd.DataFrame([summary_data])
    summary_path = os.path.join(config.OUT_TABLES, "modeling_data_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    config.log(f"모델링 데이터 요약 기록 저장: {summary_path}")

    # 검증 완료 데이터를 처리(processed) 디렉터리에 전달
    valid_path = os.path.join(config.DATA_PROCESSED, "modeling_data_validated.csv")
    df.to_csv(valid_path, index=False, encoding="utf-8-sig")
    config.log(f"검증 완료 데이터 전달 완료: {valid_path}")
    
if __name__ == "__main__":
    main()
