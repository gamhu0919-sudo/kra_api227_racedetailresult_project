import os
import pandas as pd
import numpy as np
import inference_config as cfg

def generate_relative_features(df):
    """경주 내 상대 피처 생성 (Stage03 정합성 유지)"""
    rules = {
        # (컬럼명, 원천컬럼, ascending=False 면 높은게 1등)
        "rating_rank_in_race": ("pthrRatg", False),
        "rating_zscore_in_race": ("pthrRatg", "zscore"),
        "weight_rank_in_race": ("pthrBurdWgt", False),
        "weight_zscore_in_race": ("pthrBurdWgt", "zscore"),
        "jockey_winrate_rank_in_race": ("fe_jcky_cum_win_rate", False),
        "jockey_top3_rate_rank_in_race": ("fe_jcky_cum_top3_rate", False),
        "trainer_winrate_rank_in_race": ("fe_trar_cum_win_rate", False),
        "horse_avg_rank_rank_in_race": ("fe_horse_cum_avg_rk", True), # 낮은게 좋음(1등)
        "horse_winrate_rank_in_race": ("fe_horse_cum_win_rate", False),
        "horse_experience_rank_in_race": ("fe_horse_race_count", False),
    }

    for col, args in rules.items():
        base_col = args[0]
        mode = args[1]
        
        if mode == "zscore":
            df[col] = df.groupby("race_id")[base_col].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            ).fillna(0.0)
        else:
            # Rank
            df[col] = df.groupby("race_id")[base_col].transform(
                lambda x: x.rank(method="min", ascending=mode)
            )
            
    # 특수: 레이팅 퍼센트 랭크
    df["rating_pct_rank_in_race"] = df.groupby("race_id")["pthrRatg"].transform(
        lambda x: x.rank(method="average", ascending=False, pct=True)
    )
    
    return df

def main():
    cfg.log("상대 피처 생성을 시작합니다.")
    df = pd.read_csv(cfg.PATH_FE_BASE, dtype={'hrmJckyId': str, 'hrmTrarId': str, 'pthrHrno': str}, encoding="utf-8-sig")

    # 상대 피처 생성
    df = generate_relative_features(df)

    # 저장
    df.to_csv(cfg.PATH_FE_RELATIVE, index=False, encoding="utf-8-sig")
    cfg.log(f"상대 피처 데이터 저장 완료: {cfg.PATH_FE_RELATIVE}")

if __name__ == "__main__":
    main()
