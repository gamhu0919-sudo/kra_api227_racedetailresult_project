# kra_api227_racedetailresult_project/src/xml_parser.py

import pandas as pd
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

class KRA_XMLParser:
    """한국마사회 XML 응답 파싱 및 DataFrame 변환 모듈"""
    def __init__(self, logger=None):
        self.logger = logger

    def parse_items(self, xml_text: str) -> List[Dict]:
        """XML에서 <item> 리스트 추출하여 Dict 리스트로 반환"""
        items = []
        if not xml_text:
            return items
        
        try:
            root = ET.fromstring(xml_text)
            
            # header 체크
            header = root.find("header")
            if header is not None:
                result_code = header.findtext("resultCode")
                result_msg = header.findtext("resultMsg")
                if result_code != "00":
                    if self.logger:
                        self.logger.warning(f"API Response Header (code {result_code}): {result_msg}")
                    return items
            
            # items/item 파싱
            body = root.find("body")
            if body is not None:
                total_count = body.findtext("totalCount")
                if total_count and int(total_count) == 0:
                    return items
                
                items_node = body.find("items")
                if items_node is not None:
                    for item in items_node.findall("item"):
                        item_dict = {}
                        for child in item:
                            item_dict[child.tag] = child.text
                        items.append(item_dict)
            
        except ET.ParseError as e:
            if self.logger:
                self.logger.error(f"XML Parse Error: {str(e)}")
        
        return items

    def to_dataframe(self, items: List[Dict]) -> pd.DataFrame:
        """Dict 리스트를 DataFrame으로 변환"""
        if not items:
            return pd.DataFrame()
        return pd.DataFrame(items)

    def merge_raw_files_to_interim(self, file_paths: List[str]) -> pd.DataFrame:
        """저장된 여러 XML 파일을 파싱하여 하나의 중간 DataFrame으로 병합"""
        all_items = []
        for path in file_paths:
            with open(path, 'r', encoding='utf-8') as f:
                xml_text = f.read()
                items = self.parse_items(xml_text)
                all_items.extend(items)
        
        return self.to_dataframe(all_items)
