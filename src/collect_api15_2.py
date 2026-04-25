import os
import json
import time
from api15_2_client import API15_2Client

def main():
    print("=== 한국마사회 경주마 성적 정보 (API15_2) 수집 시작 ===")
    
    # 저장 디렉토리 (raw) 확보
    raw_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'api15_2')
    os.makedirs(raw_dir, exist_ok=True)
    
    client = API15_2Client()
    
    # 파라미터는 분석기간인 2025~2026에 포커스할수 있도록 통산정보를 다가져옵니다.
    current_page = 1
    num_of_rows = 1000  # 한 번에 1천 건씩 처리 (공공데이터포털 정책 고려)
    
    total_count = None
    fetched_count = 0
    
    while True:
        print(f"Fetching page {current_page}...")
        data = client.fetch_page(page_no=current_page, num_of_rows=num_of_rows)
        
        # Fallback or Exit
        if not data:
            print(f"No Data or Error on page {current_page}. Stopping.")
            break
            
        header = data.get('response', {}).get('header', {})
        if header.get('resultCode') != '00':
            print(f"API Error: {header.get('resultMsg')} (Code: {header.get('resultCode')})")
            break
            
        body = data.get('response', {}).get('body', {})
        items = body.get('items', {})
        
        if not items:
            print(f"No items found on page {current_page}. Ending loop.")
            break
            
        # JSON 원본 물리 저장 (원형 유지)
        save_path = os.path.join(raw_dir, f"raw_api15_2_page_{current_page:03d}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        item_list = items.get('item', [])
        if isinstance(item_list, dict): 
            item_list = [item_list]
            
        current_batch_size = len(item_list)
        fetched_count += current_batch_size
        
        # 첫 페이지에서 Total Count 설정
        if total_count is None:
            total_count = body.get('totalCount', 0)
            print(f"-> Total records expected: {total_count}")
            
        print(f"-> Page {current_page} saved. Rows: {current_batch_size}")
        
        if fetched_count >= total_count:
            print("Finished fetching all available records.")
            break
            
        # 페이지 스텝
        current_page += 1
        time.sleep(0.5)  # Throttle limits

if __name__ == "__main__":
    main()
