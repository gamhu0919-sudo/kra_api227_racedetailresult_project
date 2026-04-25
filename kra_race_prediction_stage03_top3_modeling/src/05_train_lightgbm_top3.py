"""
05_train_lightgbm_top3.py
──────────────────────────────
LightGBM Top3 예측 확률 산출 모델 (베이스라인)
"""

import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

config.log_step(5, "LightGBM Binary Classifier 학습 및 예측")

def main():
    in_path = os.path.join(config.DATA_MODELING, "modeling_data_ready.csv")
    df = pd.read_csv(in_path, encoding="utf-8-sig", low_memory=False)

    # 1. 메타 컬럼 분리
    meta_cols = ['race_id', 'schdRaceDt', 'pthrHrno', 'split_group']
    
    # 2. 피처 컬럼 식별
    # 메타 및 Leakage/Target 컬럼 제외
    exclude_cols = meta_cols + config.LEAKAGE_COLS + [
        "is_valid_race", "is_leakage_free", "train_test_split_group_candidate"
    ]
    features = [c for c in df.columns if c not in exclude_cols]

    # 범주형 수동 형변환 (LightGBM 호환을 위해)
    cat_features = []
    for f in features:
        if not pd.api.types.is_numeric_dtype(df[f]) or df[f].dtype.name == 'category':
            df[f] = df[f].astype('category')
            cat_features.append(f)

    config.log(f"선택된 활용 피처 수: {len(features)}개")
    config.log(f"Categorical 피처 수: {len(cat_features)}개 -> {cat_features}")

    # 3. 데이터 분할
    train_idx = df['split_group'] == 'train'
    valid_idx = df['split_group'] == 'valid'
    test_idx  = df['split_group'] == 'test'

    X_train = df.loc[train_idx, features]
    y_train = df.loc[train_idx, config.TARGET_COL]
    
    X_valid = df.loc[valid_idx, features]
    y_valid = df.loc[valid_idx, config.TARGET_COL]

    config.log(f"Train Shape: {X_train.shape}, Valid Shape: {X_valid.shape}")

    # 4. 모델 설정 및 학습
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        objective='binary',
        importance_type='gain'
    )

    # Cat 튜닝은 LGBM의 자체 기능에 의존
    config.log("학습 시작 (Early Stopping 50)")
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='auc',
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
    )

    best_iter = lgb_model.best_iteration_
    config.log(f"학습 완료! 최적 Iteration: {best_iter}")

    # 모델 저장
    model_path = os.path.join(config.MODELS_DIR, "lightgbm_top3_baseline.pkl")
    joblib.dump(lgb_model, model_path)
    config.log(f"LightGBM 모델 저장 완료: {model_path}")

    # 피처 중요도 추출 및 저장
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    fi_path = os.path.join(config.OUT_TABLES, "lightgbm_feature_importance.csv")
    importance_df.to_csv(fi_path, index=False, encoding="utf-8-sig")

    # 5. 전체 데이터에 대한 예측
    df['pred_top3_prob'] = lgb_model.predict_proba(df[features])[:, 1]
    
    # 각 race_id 내에서 확률 상위 3위를 선발
    df = df.sort_values(['race_id', 'pred_top3_prob'], ascending=[True, False])
    df['pred_rank_in_race'] = df.groupby('race_id').cumcount() + 1
    df['pred_is_top3'] = (df['pred_rank_in_race'] <= 3).astype(int)

    # 6. 예측 결과 저장 (평가를 위해)
    output_cols = ['race_id', 'schdRaceDt', 'pthrHrno', 'split_group', 
                   config.TARGET_COL, 'pred_top3_prob', 'pred_rank_in_race', 'pred_is_top3']
    out_df = df[output_cols]
    
    pred_path = os.path.join(config.DATA_PREDICTIONS, "lightgbm_top3_predictions.csv")
    out_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    config.log(f"LightGBM 예측값 저장 완료: {pred_path}")

if __name__ == "__main__":
    main()
