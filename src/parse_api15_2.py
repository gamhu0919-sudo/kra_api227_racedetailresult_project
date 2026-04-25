import os
import json
import glob
import pandas as pd

def main():
    print("=== API15_2 원본 JSON 데이터 파싱 작업 시작 ===")
    
    # 1. 경로 설정
    raw_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'api15_2')
    interim_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'interim', 'api15_2')
    os.makedirs(interim_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(raw_dir, "raw_api15_2_page_*.json"))
    if not json_files:
        print("파싱할 Raw Data가 없습니다. collect_api15_2.py 부터 실행하세요.")
        return
        
    all_items = []
    
    # 2. JSON Extract & Flatten
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        body = data.get('response', {}).get('body', {})
        items = body.get('items', {}).get('item', [])
        
        # 만약 단일 객체면 리스트화
        if isinstance(items, dict):
            items = [items]
            
        all_items.extend(items)
        
    print(f"단순 적재 완료 - 총 {len(all_items)} Record(s)")

    # 3. CSV 변환
    if all_items:
        df = pd.DataFrame(all_items)
        
        output_csv = os.path.join(interim_dir, "parsed_api15_2.csv")
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"Interim 데이터 파싱 완료: {output_csv} (Columns: {len(df.columns)})")
    else:
        print("유효한 Item이 존재하지 않습니다.")

if __name__ == "__main__":
    main()
