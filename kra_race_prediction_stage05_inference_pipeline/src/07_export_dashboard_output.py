import os
import pandas as pd
import inference_config as cfg

def main():
    cfg.log("대시보드 연동용 데이터 수출(Export)을 시작합니다.")
    
    if not os.path.exists(cfg.PATH_PREDICTIONS):
        cfg.log("Error: 예측 결과 파일이 없습니다.")
        return

    df = pd.read_csv(cfg.PATH_PREDICTIONS, dtype={'hrmJckyId': str, 'hrmTrarId': str, 'pthrHrno': str}, encoding="utf-8-sig")

    # 1. 필수 유지 및 정렬 컬럼
    required_schema = [
        'race_id', 'schdRaceDt', 'schdRaceNo', 'pthrHrno', 'pthrHrnm', 'pthrGtno',
        'hrmJckyId', 'hrmJckyNm', 'hrmTrarId', 'hrmTrarNm', 'pthrRatg', 'pthrBurdWgt',
        'fe_horse_weight', 'fe_horse_cum_win_rate', 'fe_horse_cum_avg_rk', 'fe_horse_race_count',
        'fe_jcky_cum_win_rate', 'fe_jcky_cum_top3_rate', 'fe_trar_cum_win_rate',
        'rating_rank_in_race', 'rating_zscore_in_race', 'jockey_top3_rate_rank_in_race',
        'trainer_winrate_rank_in_race', 'horse_avg_rank_rank_in_race',
        'rsutWetr', 'rsutTrckStus', 'top3_prob', 'pred_rank', 'pred_is_top3'
    ]

    # 존재하는 컬럼만 필터링
    final_cols = [c for c in required_schema if c in df.columns]
    
    # 2. 결과 가공
    df_final = df[final_cols].copy()
    
    # 정렬: 경주별 예상 순위 순
    df_final = df_final.sort_values(['race_id', 'pred_rank'])

    # 3. 저장 (next_race_predictions.csv 덮어쓰기)
    df_final.to_csv(cfg.PATH_PREDICTIONS, index=False, encoding="utf-8-sig")
    
    # 4. 종합 요약서 생성
    report_lines = [
        "# Inference Pipeline Summary Report",
        f"- **완료 일시**: {pd.Timestamp.now()}",
        f"- **총 예측 경주 수**: {df_final['race_id'].nunique()}",
        f"- **총 예측 두수**: {len(df_final)}",
        "",
        "## 경주별 상위 3마리 요약",
    ]
    
    for rid in df_final['race_id'].unique():
        top3 = df_final[df_final['race_id'] == rid].head(3)
        report_lines.append(f"### Race ID: {rid}")
        for _, row in top3.iterrows():
            report_lines.append(f"- **{int(row['pred_rank'])}위**: {row['pthrHrnm']} (확률: {row['top3_prob']:.2%})")
        report_lines.append("")

    report_path = os.path.join(cfg.REPORTS_DIR, "inference_pipeline_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    cfg.log(f"최종 결과 저장 완료: {cfg.PATH_PREDICTIONS}")
    cfg.log(f"종합 보고서 작성 완료: {report_path}")

if __name__ == "__main__":
    main()
