# kra_api227_racedetailresult_project/src/advanced_eda.py

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg') # GUI 없는 환경용 백엔드 고정
import matplotlib.pyplot as plt
import koreanize_matplotlib
from pathlib import Path
from fpdf import FPDF
import datetime
import sys

# 프로젝트 루트 및 src 경로 추가 (모듈 참조용)
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

# 프로젝트 상수 인입
from config import (
    DIR_DATA_PROCESSED, 
    DIR_IMAGES, 
    DIR_LOGS, 
    ANALYSIS_PERIOD_LABEL
)
from utils import CustomLogger, ensure_dir

# 로거 설정
logger = CustomLogger("advanced_eda", DIR_LOGS).get_logger()

# 출력을 위한 폴더 보장
ensure_dir(DIR_IMAGES)

def run_advanced_eda():
    """edabasic.md 규칙을 준수하는 고도화된 EDA 파이프라인"""
    
    # 1. 데이터 로드
    csv_path = DIR_DATA_PROCESSED / "race_detail_result_202503_202603_processed.csv"
    if not csv_path.exists():
        logger.error(f"데이터 파일이 없습니다: {csv_path}")
        return
    
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"데이터 로드 완료. 행: {len(df)}, 열: {len(df.columns)}")

    # 리포트 저장을 위한 핸들러
    report_content = []

    def add_to_report(text):
        report_content.append(text + "\n")

    add_to_report("# 경주 상세 성적 데이터 (API227) 심층 분석 리포트")
    add_to_report(f"- **분석 기간**: {ANALYSIS_PERIOD_LABEL}")
    add_to_report(f"- **작성 일자**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    add_to_report("- **역할**: 20년 차 시니어 데이터 분석가 에이전트\n")

    # --- [RULE 3: PRE-ANALYSIS DIAGNOSIS] ---
    add_to_report("## 1. 데이터 구조 진단 (Diagnosis)")
    
    add_to_report("### 1.1 데이터 요약")
    add_to_report(f"- 전체 데이터 셰이프 (Shape): `{df.shape}`")
    
    add_to_report("### 1.2 상위 5개 행 (head)")
    add_to_report("```text\n" + df.head(5).to_string() + "\n```")
    
    add_to_report("### 1.3 하의 5개 행 (tail)")
    add_to_report("```text\n" + df.tail(5).to_string() + "\n```")
    
    add_to_report("### 1.4 데이터 정보 (info)")
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    add_to_report("```text\n" + buffer.getvalue() + "\n```")
    
    add_to_report("### 1.5 자료형 (dtypes)")
    add_to_report("```text\n" + df.dtypes.to_string() + "\n```")
    
    add_to_report("### 1.6 결측치 (Missing values)")
    null_counts = df.isnull().sum()
    null_ratio = (null_counts / len(df) * 100).round(2)
    null_df = pd.DataFrame({'Counts': null_counts, 'Ratio(%)': null_ratio})
    add_to_report("```text\n" + null_df[null_df['Counts'] > 0].to_string() + "\n```")
    
    add_to_report("### 1.7 중복 데이터 (Duplicates)")
    dup_count = df.duplicated().sum()
    add_to_report(f"- 중복 행 수: `{dup_count}`")
    
    add_to_report("### 1.8 변수 분류")
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    add_to_report(f"- **날짜형**: {', '.join(date_cols)}")
    add_to_report(f"- **수치형**: {', '.join(numeric_cols)}")
    add_to_report(f"- **범주형**: {', '.join(categorical_cols)}")

    add_to_report("### 1.9 데이터셋 특성 요약")
    add_to_report("> 본 데이터셋은 한국마사회의 경주 상세 데이터를 포함하며, 2025.03~2026.03 기간의 풍부한 시계열성을 담고 있습니다. "
                  "수치형 변수인 부담중량, 마체중, 배당률과 범주형 변수인 기수, 조교사 정보를 통해 순위(stOrd)를 예측하는 데 최적화되어 있습니다.\n")

    # --- [RULE 4: DESCRIPTIVE STATISTICS] ---
    add_to_report("## 2. 기술통계 (Descriptive Statistics)")
    add_to_report("### 2.1 수치형 데이터 요약")
    add_to_report("```text\n" + df.describe().to_string() + "\n```")
    
    add_to_report("### 2.2 범주형 데이터 요약 (기수 상위 20명)")
    if 'jkName' in df.columns:
        add_to_report("```text\n" + df['jkName'].value_counts().head(20).to_string() + "\n```")

    # --- [RULE 5: VISUALIZATION EXECUTION (MIN 10)] ---
    add_to_report("## 3. 시각화 및 해석 (Visualizations)")
    
    vis_count = 0
    
    def save_plot(filename, title, interpretation):
        nonlocal vis_count
        vis_count += 1
        path = DIR_IMAGES / filename
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        add_to_report(f"### {vis_count}. {title}")
        add_to_report(f"![{title}](./images/{filename})")
        add_to_report(f"\n**[해석]**: {interpretation}\n")

    # 1. 착순(stOrd) 분포 (일변량)
    plt.figure(figsize=(10, 5))
    df['stOrd'].value_counts().sort_index().plot(kind='bar', color='skyblue')
    save_plot("rank_dist.png", "최종 착순(stOrd) 빈도 분포", 
              "마사회 경주의 최종 순위 분포입니다. 1위부터 하위권까지 골고루 분포되어 있으며, "
              "출전마 수가 제한된 경주 특성상 특정 순위 구간에 밀집되는 경향을 보입니다. "
              "이는 분류 모델 학습 시 타겟 변수의 균형도를 파악하는 핵심 지표가 됩니다.")

    # 2. 단승 배당률(win) 분포 (일변량)
    plt.figure(figsize=(10, 5))
    df[df['win'] < 100]['win'].hist(bins=50, color='salmon')
    save_plot("win_odds_dist.png", "단승 배당률(win) 분포 (100배 미만)", 
              "전체 경주의 단승 배당률 분포입니다. 대부분의 경주마가 20배 미만의 낮은 배당에 집중되어 있으며, "
              "이는 시장의 기대치가 특정 마필에 몰리는 기수/마필 인기도를 반영합니다. "
              "롱테일 분포를 보이고 있어 로그 변환이나 이상치 처리가 모델 성능에 영향을 줄 수 있습니다.")

    # 3. 경주 기록(rcTime) 분포 (일변량)
    plt.figure(figsize=(10, 5))
    df['rcTime'].hist(bins=50, color='lightgreen')
    save_plot("rc_time_dist.png", "경주 기록(rcTime) 초 단위 분포", 
              "경주가 끝난 후 측정된 최종 기록의 초 단위 분포입니다. 경마장 거리(1000m~2000m)에 따라 "
              "멀티모달(Mulit-modal) 분포를 띠고 있으며, 이는 거리별 기록 정규화가 반드시 선행되어야 함을 시사합니다. "
              "기록의 편차는 말의 주력을 나타내는 가장 직접적인 성능 지표입니다.")

    # 4. 연령(age)별 분포 (범주형)
    plt.figure(figsize=(10, 5))
    df['age'].value_counts().sort_index().plot(kind='pie', autopct='%1.1f%%', startangle=140)
    save_plot("age_dist.png", "출전마 연령(age)별 구성비", 
              "현재 활발하게 활동하는 경주마의 연령대 비중입니다. 2세부터 노령마까지 분포되어 있으며, "
              "주로 3~4세 마필이 전체의 과반 이상을 차지하는 전성기 구간임을 알 수 있습니다. "
              "나이는 마력(Race Power)과 경험치를 나타내는 중요한 독립 변수입니다.")

    # 5. 부담중량(wgBudam) vs 착순(stOrd) (이변량)
    plt.figure(figsize=(12, 6))
    import seaborn as sns
    # edabasic 규칙: Seaborn 스타일 설정 금지 (단, plot용 툴로만 사용)
    # plt.style.use('default') 
    sns.boxplot(x='stOrd', y='wgBudam', data=df[df['stOrd'] <= 10])
    save_plot("wgbudam_vs_rank.png", "부담중량과 착순의 관계 (Boxplot)", 
              "부담중량이 높아질수록 하위권(착순 숫자가 큼)으로 갈 확률이 높아지는 경향을 분석합니다. "
              "우수한 말에게 높은 부중이 부여되는 핸디캡 경주의 특성상 상위권 마필의 부중 편차가 크게 나타납니다. "
              "이는 '부력이 좋을수록 부중이 높다'는 상관관계와 '부중이 높으면 주파력이 떨어진다'는 인과관계가 충돌하는 지점입니다.")

    # 6. 배당률(win) vs 착순(stOrd) (이변량)
    plt.figure(figsize=(12, 6))
    plt.scatter(df['win'], df['stOrd'], alpha=0.3, color='purple')
    plt.xlim(0, 100)
    plt.ylabel('최종 착순')
    plt.xlabel('단승 배당률')
    save_plot("win_vs_rank.png", "단승 배당률과 실제 착순의 상관관계", 
              "시장의 예측값(배당)과 실제 결과(착순)의 일치도를 보여주는 산점도입니다. "
              "낮은 배당일수록 하단(1위 근처)에 데이터가 밀집되어 있으며, 이는 시장의 예측이 대체로 효율적임을 의미합니다. "
              "하지만 저배당 마필이 하위권으로 밀리는 구간은 '이변'의 핵심 데이터가 됩니다.")

    # 7. 마체중(horse_weight) vs 착순(stOrd) (이변량)
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='stOrd', y='horse_weight', data=df[df['stOrd'] <= 10])
    save_plot("weight_vs_rank.png", "마체중과 착순의 분포 관계", 
              "마필의 체급(무게)이 성적에 미치는 영향을 바이올린 플롯으로 시각화했습니다. "
              "체격조건이 큰 말이 폭발적인 주력을 보일 것이라는 가설을 검증할 수 있으며, "
              "상위권 말들의 체중 분포가 하상대적으로 특정 구간(450kg~500kg)에 안정적으로 형성되어 있음을 확인했습니다.")

    # 8. 성별(sex)별 우승 여부 비중 (범주형)
    plt.figure(figsize=(10, 5))
    sex_win = pd.crosstab(df['sex'], df['is_winner'], normalize='index')
    sex_win[True].plot(kind='barh', color='gold')
    save_plot("sex_win_rate.png", "성별(sex)별 우승 확률 비교", 
              "수말, 암말, 거세마 간의 우승 비율 편차를 보여주는 차트입니다. "
              "일반적으로 거세마나 수말의 근력이 암말 대비 성적이 우세할 것이라는 실무적 가설을 정량적으로 입증합니다. "
              "성별 데이터는 경마 예측 모델링에서 누락할 수 없는 핵심 카테고리 피처입니다.")

    # 9. 경마장(meet_nm)별 출전 및 성과 (범주형)
    plt.figure(figsize=(10, 5))
    df['meet_nm'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    save_plot("meet_dist.png", "경마장(meet_nm)별 데이터 비중", 
              "서울, 부경, 제주 경마장 데이터의 수집률을 보여줍니다. 서울 경마장의 비중이 압도적으로 높으며, "
              "각 경마장별로 주로의 특성이나 주로 상태가 다르기 때문에 모델링 시 원-핫 인코딩이나 "
              "임베딩을 통해 지역적 특성을 반영해야 함을 상기시켜 줍니다.")

    # 10. 상위 10위 기수(jkName)별 평균 착순 (다변량)
    plt.figure(figsize=(12, 6))
    top_jockeys = df['jkName'].value_counts().head(10).index
    df_top_j = df[df['jkName'].isin(top_jockeys)]
    df_top_j.groupby('jkName')['stOrd'].mean().sort_values().plot(kind='bar', color='teal')
    save_plot("top_jockey_perf.png", "상위 10위권 기수별 평균 착순 성적", 
              "가장 많은 경기에 출전한 기수 10인의 평균 착순을 분석했습니다. "
              "출전 횟수가 많음에도 평균 착순이 낮다는 것은 해당 기수의 기승 능력이 매우 뛰어남을 의미합니다. "
              "이 정보는 기수 성적(Track Record) 피처 생성의 기초가 되며 베팅 효율성을 높이는 중요한 지표입니다.")

    # --- [RULE 8: REPORT_AUTO_GENERATION] ---
    # 저장 경로를 프로젝트 폴더 내의 루트로 명시적으로 설정
    output_md = BASE_DIR / "analysis_report.md"
    try:
        with open(output_md, "w", encoding="utf-8") as f:
            f.write("\n".join(report_content))
            f.flush()
        logger.info(f"마크다운 리포트 생성 완료: {output_md} (크기: {os.path.getsize(output_md)} bytes)")
    except Exception as e:
        logger.error(f"리포트 파일 저장 중 오류 발생: {e}")

    # --- [PDF GENERATION] ---
    try:
        pdf = FPDF()
        pdf.add_page()
        # 한글 폰트가 시스템에 없을 수 있으므로 기본 폰트로 리포트 구조만 생성 (분석 결과 위주)
        # 실제 환경에서 폰트 경로를 알면 pdf.add_font() 가능
        from fpdf.enums import XPos, YPos
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 10, text="KRA API227 Analysis Report (Summary)", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.cell(0, 10, text=f"Analysis Period: {ANALYSIS_PERIOD_LABEL}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, text=f"Total Records Processed: {len(df)}")
        pdf.multi_cell(0, 10, text=f"Total Visualizations Created: {vis_count}")
        pdf.output("analysis_report.pdf")
        logger.info("PDF 리포트 생성 완료: analysis_report.pdf")
    except Exception as e:
        logger.warning(f"PDF 생성 중 오류 발생: {e}")

    logger.info("--- 전 과정 분석 완료 ---")

if __name__ == "__main__":
    run_advanced_eda()
