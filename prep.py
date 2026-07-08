import pandas as pd
from pathlib import Path
import re

root_dir = Path("rome_static_medium/sched0/tr0")

def slices():
    dfs = []

    for i in range(1, 5 + 1):
        for j in range(1, 7+1):
            csv_files = list(root_dir.rglob(f"exp{i}/bs{j}/slices_bs{j}/*.csv"))

            if not csv_files:
                print(f"No CSV files found for i={i}")
                continue

            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    dfs.append(df)
                    print(f"Loaded: {csv_file}")
                except Exception as e:
                    print(f"Failed to read {csv_file}: {e}")

        if not dfs:
            print("No CSV files loaded.")
            return None

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df["Timestamp"] = merged_df.rename(columns={"Timestamp": "time"}, inplace=True)
    merged_df = merged_df.drop("Timestamp", axis=1)

    return merged_df

def ue():
    dfs = []

    attack_0 = {3, 6, 10, 13, 17, 20, 24, 27, 31, 34, 38, 41}
    attack_1 = {4, 7, 11, 14, 18, 21, 25, 28, 32, 35, 39, 42}
    attack_2 = {2, 5, 9, 12, 16, 19, 23, 26, 30, 33, 37, 40}

    def get_attack_label(ue_num):
        if ue_num in attack_0:
            return 0
        elif ue_num in attack_1:
            return 1
        else:
            return 2

    for i in range(1, 5 + 1):
        for j in range(1, 7 + 1):
            csv_files = list(root_dir.rglob(f"exp{i}/bs{j}/ue*.csv"))

            if not csv_files:
                print(f"No CSV files found for i={i}")
                continue

            for csv_file in csv_files:
                try:
                    match = re.search(r"ue(\d+)", csv_file.stem)
                    ue_num = int(match.group(1)) if match else None

                    df = pd.read_csv(csv_file)
                    df["attack"] = get_attack_label(ue_num)
                    dfs.append(df)
                    print(f"Loaded: {csv_file}")
                except Exception as e:
                    print(f"Failed to read {csv_file}: {e}")

        if not dfs:
            print("No CSV files loaded.")
            return None

    merged_df = pd.concat(dfs, ignore_index=True)

    return merged_df

def bs():
    dfs = []

    for i in range(1, 5 + 1):
        for j in range(1, 7 + 1):
            csv_files = list(root_dir.rglob(f"exp{i}/bs{j}/bs{j}.csv"))

            if not csv_files:
                print(f"No CSV files found for i={i}")
                continue

            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    dfs.append(df)
                    print(f"Loaded: {csv_file}")
                except Exception as e:
                    print(f"Failed to read {csv_file}: {e}")

        if not dfs:
            print("No CSV files loaded.")
            return None

    merged_df = pd.concat(dfs, ignore_index=True)

    return merged_df

def concat_csvs():
    slices_csvs = slices()
    ue_csvs = ue()
    bs_csvs = bs()
    df1 = pd.merge_asof(
        ue_csvs.sort_values('time'),
        bs_csvs.sort_values('time'),
        on='time',
        suffixes=('_ue', '_bs'),
        direction='nearest'
    )
    df = pd.merge_asof(
        df1.sort_values('time'),
        slices_csvs.sort_values("time"),
        on="time",
        suffixes=('_ns', "_s"),
        direction="nearest"
    )

    df.to_csv("all.csv", index=False)
    return df