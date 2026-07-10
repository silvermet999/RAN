import pandas as pd
# from data_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler


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
X_train_cont = scaler.fit_transform(X_train_cont)
X_test_cont = scaler.transform(X_test_cont)

