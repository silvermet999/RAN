import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp, entropy

import prep
import prep_OOD


def js_divergence(a, b, bins: int = 50):
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if lo == hi:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)

    p, _ = np.histogram(a, bins=edges, density=True)
    q, _ = np.histogram(b, bins=edges, density=True)

    eps = 1e-12
    p = p / (p.sum() + eps) + eps
    q = q / (q.sum() + eps) + eps
    m = 0.5 * (p + q)

    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)


def compute_feature_distances(df_a, df_b, features=None):
    if features is None:
        numeric_a = df_a.select_dtypes(include=[np.number]).columns
        numeric_b = df_b.select_dtypes(include=[np.number]).columns
        features = sorted(set(numeric_a) & set(numeric_b))

    rows = []
    for feat in features:
        a = df_a[feat].dropna().to_numpy()
        b = df_b[feat].dropna().to_numpy()

        if len(a) == 0 or len(b) == 0:
            continue

        pooled_std = np.std(np.concatenate([a, b]))
        pooled_std = pooled_std if pooled_std > 0 else 1.0

        w_dist_raw = wasserstein_distance(a, b)
        w_dist_norm = w_dist_raw / pooled_std

        ks_stat, ks_pval = ks_2samp(a, b)
        jsd = js_divergence(a, b)

        rows.append({
            "feature": feat,
            "wasserstein_raw": w_dist_raw,
            "wasserstein_normalized": w_dist_norm,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pval,
            "js_divergence": jsd,
            "mean_a": a.mean(),
            "mean_b": b.mean(),
            "std_a": a.std(),
            "std_b": b.std(),
            "n_a": len(a),
            "n_b": len(b),
        })

    result = pd.DataFrame(rows).sort_values("wasserstein_normalized", ascending=False)
    result = result.reset_index(drop=True)
    return result


def flag_discriminative_features(dist_df, ks_alpha= 0.05, w_norm_threshold = 0.2):
    df = dist_df.copy()
    df["discriminative"] = (
        (df["ks_pvalue"] < ks_alpha) & (df["wasserstein_normalized"] > w_norm_threshold)
    )
    return df

dist_df = compute_feature_distances(prep.df, prep_OOD.df_sched1)
dist_df = flag_discriminative_features(dist_df)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)
print(dist_df[[
    "feature", "wasserstein_normalized", "ks_statistic",
    "ks_pvalue", "js_divergence", "discriminative"
]])


# PF, WF: discriminative
# False    33
# True     17

# PF, RR: discriminative
# False    32
# True     18

# RR, WF: discriminative
# False    42
# True     16
