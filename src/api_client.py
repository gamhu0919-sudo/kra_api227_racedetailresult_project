# kra_api227_racedetailresult_project/src/api_client.py

import requests
import time
from typing import Optional, Dict

class KRA_APIClient:
    """한국마사회 API 호출 클라이언트 (XML 전용)"""
    def __init__(self, base_url: str, service_key: str, logger=None):
        self.base_url = base_url
        self.service_key = service_key
        self.logger = logger

    def fetch_xml(self, meet: int, rc_date: str, rc_no: int, page_no: int = 1, num_rows: int = 100) -> Optional[str]:
        """API를 호출하고 XML 문자열을 반환"""
        params = {
            "serviceKey": self.service_key,
            "pageNo": str(page_no),
            "numOfRows": str(num_rows),
            "meet": str(meet),
            "rc_date": rc_date,
            "rc_no": str(rc_no)
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            status_code = response.status_code
            
            if status_code == 200:
                return response.text
            else:
                if self.logger:
                    self.logger.error(f"API Error (HTTP {status_code}): {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            if self.logger:
                self.logger.error(f"Request Exception: {str(e)}")
            return None

    def test_call(self, meet: int = 1, rc_date: str = "20250301", rc_no: int = 1):
        """단건 테스트 호출용 메서드"""
        if self.logger:
            self.logger.info(f"Test Call: meet={meet}, date={rc_date}, no={rc_no}")
        xml_res = self.fetch_xml(meet, rc_date, rc_no)
        if xml_res:
            if self.logger:
                self.logger.info("Test Success. Raw XML Sample (1000 chars):")
                self.logger.info(xml_res[:1000])
        else:
            if self.logger:
                self.logger.error("Test Failed.")
        return xml_res
