import subprocess
import sys
import os

def run_script(script_path):
    print(f"\n[Stage 05] >>> Running: {script_path}")
    # src 폴더 내부에서 실행되도록 Cwd 조정 또는 경로 전달
    # 이번에는 스크립트 내에서 상대 경로 import(00_config)를 사용하므로 src 폴더에서 실행하는 것이 안전
    script_dir = os.path.dirname(script_path)
    script_name = os.path.basename(script_path)
    
    result = subprocess.run([sys.executable, script_name], cwd=script_dir, capture_output=False)
    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} 실행 중 오류가 발생했습니다.")
        sys.exit(1)

def main():
    base_path = "kra_race_prediction_stage05_inference_pipeline/src"
    
    steps = [
        "01_create_reference_tables.py",
        "02_create_next_race_template.py",
        "03_validate_next_race_input.py",
        "04_build_inference_features.py",
        "05_make_relative_features.py",
        "06_predict_next_race.py",
        "07_export_dashboard_output.py"
    ]
    
    print("="*60)
    print("Stage 05 Inference Pipeline Stability Start")
    print("="*60)

    for step in steps:
        run_script(os.path.join(base_path, step))
    
    print("\n" + "="*60)
    print("[SUCCESS] Stage 05 Inference Pipeline Hardening Completed!")
    print("="*60)

if __name__ == "__main__":
    main()
