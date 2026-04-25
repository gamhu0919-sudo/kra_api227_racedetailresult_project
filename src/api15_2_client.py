import os
import requests
import urllib.parse
import time
from dotenv import load_dotenv

class API15_2Client:
    """한국마사회 경주마 성적 정보 API (API15_2) 통신 전담 클래스.
    원칙: 기존 API 연결 환경설정(.env) 로직을 계승하며, 안정적인 재시도 로직을 포함.
    """
    def __init__(self):
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        load_dotenv(env_path)
        
        # ServiceKey 인코딩 이슈를 막기 위해 미리 unquote 처리하여 관리.
        raw_key = os.getenv("KRA_SERVICE_KEY") or os.getenv("SERVICE_KEY") or '4ff4f09d885da1194ab2020ae8bc356906b16a8222a30285fdcbf9efb32fd97e'
        self.service_key = urllib.parse.unquote(raw_key)
        
        self.base_url = "https://apis.data.go.kr/B551015/API15_2"
        
    def fetch_page(self, page_no=1, num_of_rows=1000, max_retries=3):
        """특정 페이지 번호에 대한 JSON 객체를 반환합니다."""
        params = {
            "ServiceKey": self.service_key,
            "pageNo": str(page_no),
            "numOfRows": str(num_of_rows),
            "_type": "json"
        }
        
        for attempt in range(max_retries):
            try:
                res = requests.get(self.base_url, params=params, timeout=10)
                if res.status_code == 200:
                    try:
                        return res.json()
                    except ValueError:
                        # XML 강제 캐스팅이 터졌거나 기타 문자열 에러 발생시
                        print(f"[!] JSON parsing error on page {page_no}.")
                        return None
                else:
                    print(f"[-] HTTP Error {res.status_code} on page {page_no}. Retrying...")
            except requests.exceptions.RequestException as e:
                print(f"[-] Request failed on page {page_no}: {e}")
                
            time.sleep(2)
        
        print(f"[x] Failed to fetch page {page_no} after {max_retries} attempts.")
        return None
