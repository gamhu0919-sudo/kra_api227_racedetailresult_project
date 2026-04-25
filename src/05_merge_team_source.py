import pandas as pd
import numpy as np
import os
import glob

def clean_id(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if val_str.endswith(".0"):
        val_str = val_str[:-2]
    # Remove leading zeros to safely unify "080459" and "80459"
    # But ONLY for numeric looking IDs
    if val_str.isdigit():
        return str(int(val_str))
    return val_str

def main():
    data_dir = r"c:\Users\kang_\Desktop\NewKyungma\kra_api227_racedetailresult_project\data\team_source"
    output_dir = r"c:\Users\kang_\Desktop\NewKyungma\kra_api227_racedetailresult_project\data\processed"
    os.makedirs(output_dir, exist_ok=True)
    
    print("1. Loading files...")
    # Base tables
    df_1 = pd.read_csv(os.path.join(data_dir, "1_경주성적정보(15063979).csv"), encoding="utf-8", dtype=str)
    
    # Jockey tables
    df_j2 = pd.read_csv(os.path.join(data_dir, "2_기수성적정보_Jeju.csv"), encoding="utf-8", dtype=str)
    df_j3_1 = pd.read_csv(os.path.join(data_dir, "3_기수성적정보_Busan.csv"), encoding="utf-8", dtype=str)
    df_j3_2 = pd.read_csv(os.path.join(data_dir, "3_기수성적정보_Seoul.csv"), encoding="utf-8", dtype=str)
    
    print("2. Merging Jockey files...")
    df_jockey = pd.concat([df_j2, df_j3_1, df_j3_2], ignore_index=True)
    df_jockey["jkNo"] = df_jockey["jkNo"].apply(clean_id)
    # Deduplicate jockey data using jkNo (assuming latest or same)
    df_jockey = df_jockey.drop_duplicates(subset=["jkNo"], keep="last")
    print(f"  -> Merged jockey rows: {len(df_jockey)}")
    
    print("3. Preprocessing Main Table (1번 파일)")
    # We will use df_1 as the base master table.
    df_main = df_1.copy()
    df_main["jkNo"] = df_main["jkNo"].apply(clean_id)
    df_main["trNo"] = df_main["trNo"].apply(clean_id)
    df_main["hrNo"] = df_main["hrNo"].apply(clean_id)
    df_main["rcDate"] = df_main["rcDate"].apply(clean_id)
    
    print("4. Joining Master with Jockey...")
    # Add prefix to jockey columns to avoid collision (e.g. meet)
    jockey_cols_to_use = df_jockey.columns.difference(['jkName']) # Keep jkNo for joining
    df_j_subset = df_jockey[jockey_cols_to_use].rename(columns={c: f"jockey_stats_{c}" for c in jockey_cols_to_use if c != "jkNo"})
    
    df_merged = pd.merge(df_main, df_j_subset, on="jkNo", how="left")
    missing_jockey = df_merged["jockey_stats_winRateT"].isna().sum()
    print(f"  -> Jockey matching rate: {(len(df_merged) - missing_jockey) / len(df_merged) * 100:.2f}% (Missing: {missing_jockey} rows)")
    
    print("5. Looking into Detailed Result tables (7, 8, 9)")
    df_7 = pd.read_csv(os.path.join(data_dir, "7_경주별상세성적표.csv"), encoding="utf-8", dtype=str)
    df_8 = pd.read_csv(os.path.join(data_dir, "8_경주별상세성적표.csv"), encoding="utf-8", dtype=str)
    
    try:
        df_9 = pd.read_csv(os.path.join(data_dir, "9_경주별상세성적표.csv"), encoding="utf-8", dtype=str)
    except UnicodeDecodeError:
        try:
            df_9 = pd.read_csv(os.path.join(data_dir, "9_경주별상세성적표.csv"), encoding="cp949", dtype=str)
        except UnicodeDecodeError:
            df_9 = pd.read_csv(os.path.join(data_dir, "9_경주별상세성적표.csv"), encoding="euc-kr", dtype=str)
            
    df_details = pd.concat([df_7, df_8, df_9], ignore_index=True)
    df_details["hrNo"] = df_details["hrNo"].apply(clean_id)
    df_details["jkNo"] = df_details["jkNo"].apply(clean_id)
    df_details["rcDate"] = df_details["rcDate"].apply(clean_id)
    
    # drop duplicates
    subset_keys = ["hrNo", "jkNo", "rcDate"]
    df_details = df_details.drop_duplicates(subset=subset_keys, keep="last")
    print(f"  -> Details table deduplicated: {len(df_details)} rows remain from ({len(df_7)+len(df_8)+len(df_9)} total)")
    
    # Instead of full merge which might explode columns, 
    # we just provide the processed main merged list and details list separately if they represent different granularity,
    # OR we merge them. The user wants them merged. 
    # Let's attach some columns from df_details to df_merged if hrNo, jkNo, rcDate match.
    # To avoid duplicated columns, we only take those not in df_merged
    overlap_cols = list(set(df_merged.columns) & set(df_details.columns))
    overlap_cols.remove("hrNo")
    overlap_cols.remove("jkNo")
    overlap_cols.remove("rcDate")
    # Also remove some other potential keys so we don't drop them
    if "chulNo" in overlap_cols: overlap_cols.remove("chulNo")
        
    df_details_subset = df_details.drop(columns=overlap_cols, errors='ignore')
    
    print("6. Final Join Main + Details")
    df_final = pd.merge(df_merged, df_details_subset, on=["hrNo", "jkNo", "rcDate", "chulNo"], how="left")
    
    print(f"  -> Final Row Count: {len(df_final)} (Base was {len(df_main)})")
    
    out_file = os.path.join(output_dir, "merged_team_source.csv")
    df_final.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\nSaved successfully to {out_file}")
    
    # write matching stats to a text to report
    with open("merge_stats.txt", "w", encoding="utf-8") as f:
        f.write(f"base_rows: {len(df_main)}\n")
        f.write(f"jockey_missing: {missing_jockey}\n")
        f.write(f"details_rows: {len(df_details)}\n")
        f.write(f"final_rows: {len(df_final)}\n")

if __name__ == "__main__":
    main()
