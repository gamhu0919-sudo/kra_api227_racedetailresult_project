"""
rule_baseline.py
──────────────────────────────
단순 규칙에 따라 경주 출전마 대상 Top 3 픽스(Picks)를 반환하는 모듈
"""

import pandas as pd

def _select_top3(df_race, score_col, ascending=False):
    """지정된 점수 컬럼 기준 상위 3위를 선발 (Top 3 이내면 1, 아니면 0 인 시리즈 리턴)"""
    if len(df_race) == 0:
        return pd.Series(dtype=int)
        
    s_col_filled = df_race[score_col].fillna(0)
    
    if ascending:
        rank_s = s_col_filled.rank(method='min', ascending=True)
    else:
        rank_s = s_col_filled.rank(method='min', ascending=False)
        
    return (rank_s <= 3).astype(int)

def rating_top3_rule(df_race):
    """레이팅이 큰 상위 3마리"""
    return _select_top3(df_race, 'pthrRatg', ascending=False)

def horse_avg_rank_top3_rule(df_race):
    """과거 평균순위가 낮은(좋은) 상위 3마리"""
    # 값이 낮을수록 기록이 좋음 (예: 평균 1.5등)
    return _select_top3(df_race, 'fe_horse_cum_avg_rk', ascending=True)

def jockey_winrate_top3_rule(df_race):
    """기수 승률 상위 3마리"""
    return _select_top3(df_race, 'fe_jcky_cum_win_rate', ascending=False)

def trainer_winrate_top3_rule(df_race):
    """조교사 승률 상위 3마리"""
    return _select_top3(df_race, 'fe_trar_cum_win_rate', ascending=False)

def composite_rule(df_race):
    """EDA 과정에서 최적화된 복합 룰 상위 3마리"""
    if len(df_race) == 0:
        return pd.Series(dtype=int)
        
    df = df_race.copy()
    
    rt_score = df['pthrRatg'].rank(pct=True, ascending=True).fillna(0.5)
    hr_score = df['fe_horse_cum_avg_rk'].rank(pct=True, ascending=False).fillna(0.5) 
    jw_score = df['fe_jcky_cum_win_rate'].rank(pct=True, ascending=True).fillna(0.5)
    tw_score = df['fe_trar_cum_win_rate'].rank(pct=True, ascending=True).fillna(0.5)
    pw_score = df['pthrBurdWgt'].rank(pct=True, ascending=True).fillna(0.5)
    
    df['composite_score'] = (
        rt_score * 1.5 + 
        hr_score * 2.0 + 
        jw_score * 1.0 + 
        tw_score * 1.0 - 
        (pw_score * 0.5)
    )
    
    return _select_top3(df, 'composite_score', ascending=False)
