import os
import joblib
import pandas as pd
import numpy as np
import inference_config as cfg

def main():
    cfg.log("모델 추론(Prediction)을 시작합니다.")
    
    # 1. 모델 로드
    if not os.path.exists(cfg.PATH_MODEL):
        cfg.log(f"Error: 모델 파일이 없습니다. ({cfg.PATH_MODEL})")
        return
    
    model = joblib.load(cfg.PATH_MODEL)
    expected_features = model.feature_name_
    cfg.log(f"모델 기대 피처 수: {len(expected_features)}")

    # 2. 데이터 로드
    df = pd.read_csv(cfg.PATH_FE_RELATIVE, dtype={'hrmJckyId': str, 'hrmTrarId': str, 'pthrHrno': str}, encoding="utf-8-sig")

    # 3. 피처 동기화 및 누락 확인
    found_features = [f for f in expected_features if f in df.columns]
    missing_features = [f for f in expected_features if f not in df.columns]

    # 누락 피처 보완 (0.0으로 임시 할당)
    for f in missing_features:
        df[f] = 0.0

    # 4. 보고서 작성
    alignment_report = [
        "# Model Feature Alignment Report",
        f"- **일시**: {pd.Timestamp.now()}",
        f"- **모델 기대 피처 수**: {len(expected_features)}",
        f"- **발견된 피처 수**: {len(found_features)}",
        f"- **누락된 피처 수**: {len(missing_features)}",
        "",
        "## 누락된 피처 목록 (0.0 보완됨)"
    ]
    if missing_features:
        alignment_report.append("\n".join([f"- {f}" for f in missing_features]))
    else:
        alignment_report.append("- 없음 (모든 피처 일치)")

    report_path = os.path.join(cfg.REPORTS_DIR, "model_feature_alignment_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(alignment_report))

    # 5. 예측 실행
    X_inference = df[expected_features].copy()
    
    # 범주형 변환 (학습 시와 동일하게)
    for col in X_inference.columns:
        if not pd.api.types.is_numeric_dtype(X_inference[col]):
            X_inference[col] = X_inference[col].astype('category')

    # 확률 예측
    probs = model.predict_proba(X_inference)[:, 1]
    df['top3_prob'] = probs

    # 6. 경주별 순위 산정
    df['pred_rank'] = df.groupby('race_id')['top3_prob'].rank(ascending=False, method='min')
    df['pred_is_top3'] = (df['pred_rank'] <= 3).astype(int)

    # 7. 저장
    df.to_csv(cfg.PATH_PREDICTIONS, index=False, encoding="utf-8-sig")
    cfg.log(f"예측 완료 및 저장: {cfg.PATH_PREDICTIONS}")

if __name__ == "__main__":
    main()
