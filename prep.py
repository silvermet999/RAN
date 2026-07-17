import pandas as pd
from pathlib import Path
import re
# from data_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler


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


df = pd.read_csv("dataset/all.csv",index_col=False)

df = df.drop(["Unnamed: 4", "Unnamed: 10", "Unnamed: 18", "Unnamed: 28", "Unnamed: 31"], axis=1)
df = df.dropna(axis=0)
for col in df.columns:
    if len(df[col].unique()) == 1:
        df.drop(col, inplace=True, axis=1)

# df = df.drop(["slicing_enabled", "power_multiplier", "scheduling_policy", "tx_errors downlink (%)", "ul_rssi",
#               "dl_pmi", "dl_ri", "ul_n", "cc", "rf_o"], axis=1)
# "pci", "earfcn",  "ul_ta", "rf_u", "is_attached"
# report = ProfileReport(df)
# report.to_file("report.html")


corr_matrix = df.corr()
threshold = 0.75
pairs = []
cols = corr_matrix.columns
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        val = corr_matrix.iloc[i, j]
        if val > threshold or val < -threshold:
            pairs.append((cols[i], cols[j], val))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for col1, col2, val in pairs:
    print(f"{col1} <-> {col2}: {val:.3f}")

df['is_detached'] = (df['ul_ta'] == 0.0).astype(int)
df['ta_attach_diverge'] = ((df['is_attached'] == 1) & (df['ul_ta'] == 0.0)).astype(int)
df['impossible_state'] = ((df['is_attached'] == 0) & (df['ul_ta'] > 0.0)).astype(int)
df['ul_ta_tier'] = df['ul_ta'].map({0.52: 1, 1.0: 2, 2.1: 3}).fillna(0).astype(int)

df = df.drop(["pci", "is_attached", "ul_ta", "pl", "earfcn"], axis=1)
X = df.drop("attack", axis=1)
y = df["attack"]
# cat: earfcn rf_u is_attached rsrp dl_mac_ns dl_snr dl_turbo dl_bler ul_buff ul_bler rf_l
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train_cont = X_train.drop(["rf_u", "rsrp", "dl_mcs_ns", "dl_snr", "dl_turbo", "dl_bler", "ul_ta_tier"
                        , "ul_buff" ,"ul_bler" ,"rf_l", "is_detached", "ta_attach_diverge", "impossible_state"], axis=1)

X_test_cont = X_test.drop(["rf_u", "rsrp", "dl_mcs_ns", "dl_snr", "dl_turbo", "dl_bler", "ul_ta_tier"
                        , "ul_buff" ,"ul_bler" ,"rf_l", "is_detached", "ta_attach_diverge", "impossible_state"], axis=1)

X_train_disc = X_train[["rf_u", "rsrp", "dl_mcs_ns", "dl_snr", "dl_turbo", "dl_bler", "ul_ta_tier"
                        , "ul_buff" ,"ul_bler" ,"rf_l", "is_detached", "ta_attach_diverge", "impossible_state"]]

X_test_disc = X_test[["rf_u", "rsrp", "dl_mcs_ns", "dl_snr", "dl_turbo", "dl_bler", "ul_ta_tier"
                        , "ul_buff" ,"ul_bler" ,"rf_l", "is_detached", "ta_attach_diverge", "impossible_state"]]
X_train_disc = X_train_disc.astype(int)
X_test_disc = X_test_disc.astype(int)

enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_disc[X_train_disc.columns] = enc.fit_transform(X_train_disc[X_train_disc.columns]).astype(int)
X_test_disc[X_train_disc.columns] = enc.transform(X_test_disc[X_train_disc.columns]).astype(int)

scaler = MinMaxScaler(feature_range=(-1, 1))
cont_copy = X_train_cont.copy()
X_train_cont = scaler.fit_transform(X_train_cont)
X_test_cont = scaler.transform(X_test_cont)

X_train_cont_df = pd.DataFrame(X_train_cont, columns=cont_copy.columns)
X_test_cont_df = pd.DataFrame(X_test_cont, columns=cont_copy.columns)

X_train_sc = pd.concat([X_train_disc, X_train_cont_df], axis=1)
X_test_sc = pd.concat([X_test_disc, X_test_cont_df], axis=1)

