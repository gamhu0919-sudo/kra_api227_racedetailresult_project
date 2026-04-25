# API15_2 경주마 성적 정보 통합 변수사전(Data Dictionary)

본 사전에 기재된 필드들은 실제 KRA 데이터 수집 후 Pandas 전처리를 통해 정립된 컬럼 규격이며, 이 변수들을 활용하여 API227과 같은 파생 지표 분석에 징검다리 역할을 수행합니다.

| 분류 | 예상 API(JSON) 키 | 정형화 컬럼명(Processed) | 자료형 | 비고 (설명) |
|---|---|---|---|---|
| **기본 식별정보** | `meet` | `meet` | str | 시행경마장명 (예: 서울, 부경, 제주) |
| **기본 식별정보** | `hrName` | `hr_name` | str | 마명 |
| **기본 식별정보** | `hrNo` | `hr_no` | str | 마번(말고유번호) - **조인용 핵심키** |
| 기본 스펙 | `name` (또는 `sex`) | `sex` | str | 성별 (수/암/거세) |
| 기본 스펙 | `age` | `age` | int | 나이 |
| 기본 스펙 | `name2` (또는 `origin`) | `origin` | str | 산지 |
| 기본 스펙 | `debutDt` | `debut_date` | datetime | 데뷔일자 |
| **최신 기록** | `recentRcDate` | `recent_rc_date` | datetime | 최근 경주일자 (분석기간 `25년 3월~26년 3월` 필터링 활용점) |
| 최신 기록 | `recentRcNo` | `recent_rc_no` | int | 최근 경주번호 |
| 최신 기록 | `recentRank` | `recent_rank` | int | 최근 경주순위 |
| 최신 기록 | `recentRcTime` | `recent_rc_time` | float | 최근 경주기록 (초단위 변환 또는 그대로 사용) |
| 최신 기록 | `recentChulWt` | `recent_chul_wt` | float | 최근 경주부담중량 |
| 최신 기록 | `recentRating` | `recent_rating` | int | 최근 경주레이팅 |
| 최신 기록 | `recentHrWt` | `recent_hr_wt` | int | 최근 경주마체중 |
| 통산 기록 | `totalRcCnt` | `total_rc_cnt` | int | 통산 총출주회수 |
| 통산 기록 | `total1stCnt` | `total_1st_cnt` | int | 통산 1착회수 |
| 통산 기록 | `total2ndCnt` | `total_2nd_cnt` | int | 통산 2착회수 |
| 통산 기록 | `totalWinRate` | `total_win_rate` | float | 통산 승률 |
| 통산 기록 | `totalShowRate` | `total_show_rate` | float | 통산 복승률 |
| 누적 매출 | `totalPrize` | `total_prize` | int | 통산 착순상금 |
| 누적 매출 | `recent1YPriz` | `recent_1y_prize` | int | 최근 1년 착순상금 |
| 누적 매출 | `recent6MPriz` | `recent_6m_prize` | int | 최근 6개월 수득상금 |

*(주의: 위 표의 `추측된 JSON 키`는 향후 Raw Data 수집을 덤프해 본 뒤 실제 응답되는 CamelCase/PascalCase의 명칭으로 파싱 로직에서 확정할 예정임)*
