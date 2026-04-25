import os
import pandas as pd
import inference_config as cfg

def main():
    cfg.log("입력 데이터 검증(Validation)을 시작합니다.")
    
    if not os.path.exists(cfg.PATH_INPUT_ENTRIES):
        cfg.log(f"Error: 입력 파일이 없습니다. ({cfg.PATH_INPUT_ENTRIES})")
        return

    df = pd.read_csv(cfg.PATH_INPUT_ENTRIES, dtype={'hrmJckyId': str, 'hrmTrarId': str, 'pthrHrno': str}, encoding="utf-8-sig")

    # 1. 필수 컬럼 확인
    required_cols = [
        'race_id', 'schdRaceDt', 'schdRaceNo', 'pthrHrno', 'pthrHrnm', 
        'hrmJckyId', 'hrmJckyNm', 'hrmTrarId', 'hrmTrarNm', 'pthrBurdWgt', 
        'pthrGtno', 'pthrRatg', 'cndRaceClas', 'cndBurdGb', 'cndGndr', 
        'cndAg', 'cndRatg', 'fe_race_dist', 'hrmJckyAlw', 'fe_horse_weight', 
        'rsutWetr', 'rsutTrckStus', 'fe_track_humidity'
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]

    # 2. 금지 컬럼 확인 (Leakage)
    leakage_cols = [
        'rsutRk', 'target_rank', 'target_is_top3', 'rsutRaceRcd', 
        'rsutMargin', 'rsutWinPrice', 'rsutQnlaPrice', 'rsutRkPurse', 'rsutRkAdmny'
    ]
    found_leakage = [c for c in leakage_cols if c in df.columns]

    # 3. 데이터 무결성 체크
    # 중복 마번
    duplicate_horses = df[df.duplicated(subset=['race_id', 'pthrHrno'])]
    
    # race_id별 출전두수 (3두 미만 차단)
    race_counts = df.groupby('race_id').size()
    low_count_races = race_counts[race_counts < 3]

    # 4. 보고서 생성
    report_lines = [
        "# Input Validation Report",
        f"- **검증 일시**: {pd.Timestamp.now()}",
        f"- **대상 파일**: {cfg.PATH_INPUT_ENTRIES}",
        f"- **총 행 수**: {len(df)}",
        f"- **검증된 경주 수**: {len(race_counts)}",
        "",
        "## 검증 결과 요약",
    ]

    is_passed = True
    if missing_cols:
        report_lines.append(f"- ❌ **필수 컬럼 누락**: {missing_cols}")
        is_passed = False
    else:
        report_lines.append("- ✅ 모든 필수 컬럼 존재")

    if found_leakage:
        report_lines.append(f"- ❌ **Leakage 변수 감지**: {found_leakage} (미래 예측에 사용 불가)")
        is_passed = False
    else:
        report_lines.append("- ✅ Leakage 변수 없음")

    if not duplicate_horses.empty:
        report_lines.append(f"- ❌ **중복 출전마 감지**: {duplicate_horses['pthrHrno'].unique()}")
        is_passed = False
    else:
        report_lines.append("- ✅ 출전마 중복 없음")

    if not low_count_races.empty:
        report_lines.append(f"- ❌ **출전 두수 부족 경주**: {low_count_races.index.tolist()} (3두 미만)")
        is_passed = False
    else:
        report_lines.append("- ✅ 출전 두수 무결성 통과")

    report_lines.append("" if is_passed else "\n> [!CAUTION]\n> 검증에 실패한 항목이 있습니다. 데이터를 수정한 후 다시 실행하십시오.")

    # 저장
    report_path = os.path.join(cfg.REPORTS_DIR, "input_validation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    cfg.log(f"검증 보고서 작성 완료: {report_path}")
    if not is_passed:
        cfg.log("Warning: 검증 실패 항목이 발견되었습니다. 보고서를 확인하십시오.")

if __name__ == "__main__":
    main()
