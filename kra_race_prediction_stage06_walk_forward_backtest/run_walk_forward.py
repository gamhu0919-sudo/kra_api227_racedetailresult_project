"""
run_walk_forward.py
──────────────────────────────
Stage 06 Walk-forward Backtest 메인 오케스트레이터
"""

import sys
import os

# src 폴더를 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "src"))

import importlib.util

def import_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

trainer = import_module_from_path("trainer", os.path.join(current_dir, "src", "04_train_predict_monthly.py"))
reporter = import_module_from_path("reporter", os.path.join(current_dir, "src", "06_make_walk_forward_report.py"))
cfg = import_module_from_path("cfg", os.path.join(current_dir, "src", "config_walk_forward.py"))

def main():
    cfg.log("=============================================")
    cfg.log("  Stage 06: Walk-forward Backtest 시작")
    cfg.log("=============================================")
    
    # 1. 실행 및 결과 수집
    metrics_df, preds_all = trainer.run_walk_forward()
    
    if metrics_df is not None:
        cfg.log("백테스트 완료. 리포트 생성 중...")
        
        # 2. 시각화 및 리포트
        reporter.create_plots(metrics_df)
        reporter.generate_markdown_report(metrics_df)
        
        cfg.log(f"최종 결과 저장 완료: {cfg.REPORTS_DIR}")
        cfg.log("=============================================")
        cfg.log(f"평균 Precision@3: {metrics_df['precision_at_3'].mean():.2%}")
        cfg.log(f"평균 Hit@3: {metrics_df['hit_at_3'].mean():.2%}")
        cfg.log("=============================================")
    else:
        cfg.log("백테스트 실패 또는 중단됨.")

if __name__ == "__main__":
    main()
