import subprocess
import sys
import os

def run_script(script_path):
    print(f"\n>>> Running: {script_path}")
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode != 0:
        print(f"Error occurred while running {script_path}")
        sys.exit(1)

def main():
    base_path = "kra_race_prediction_stage05_inference_pipeline/src"
    
    # 1. 기준 데이터 생성 (Reference stats)
    run_script(os.path.join(base_path, "00_create_reference_data.py"))
    
    # 2. 추론 실행 (Inference Pipeline)
    # 주의: next_race_template.csv가 준비되어 있어야 함
    run_script(os.path.join(base_path, "01_inference_pipeline.py"))
    
    print("\n[SUCCESS] Inference pipeline completed successfully.")

if __name__ == "__main__":
    main()
