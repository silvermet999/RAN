import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from umap import UMAP
from sklearn.metrics.pairwise import rbf_kernel

rr = pd.read_csv("src/sched0.csv")
wf = pd.read_csv("src/sched1.csv")
pf = pd.read_csv("src/OOD.csv")

rr = rr.drop(["Unnamed: 4", "Unnamed: 10", "Unnamed: 18", "Unnamed: 28", "Unnamed: 31"], axis=1)
wf = wf.drop(["Unnamed: 4", "Unnamed: 10", "Unnamed: 18", "Unnamed: 28", "Unnamed: 31"], axis=1)
pf = pf.drop(["Unnamed: 4", "Unnamed: 10", "Unnamed: 18", "Unnamed: 28", "Unnamed: 31"], axis=1)

rr = rr.drop("attack", axis=1)
wf = wf.drop("attack", axis=1)
pf = pf.drop("attack", axis=1)
rr = rr.dropna()
wf = wf.dropna()
pf = pf.dropna()

scaler = StandardScaler()

rr_scaled = scaler.fit_transform(rr)
wf_scaled = scaler.transform(wf)
pf_scaled = scaler.transform(pf)

# X = np.vstack([rr_scaled, wf_scaled]) #, pf_scaled])
#
# labels = (
#     ["RR"] * len(rr_scaled)
#     + ["WF"] * len(wf_scaled)
#     #+ ["PF"] * len(pf_scaled)
# )
#
# umap = UMAP(
#     n_neighbors=30,
#     min_dist=0.1,
#     random_state=42
# )
#
# embedding = umap.fit_transform(X)
#
# plt.figure(figsize=(9,7))
#
# for name, colour in zip(
#     ["RR","WF"], #,"PF"],
#     ["blue","green"] #,"red"]
# ):
#     idx = np.array(labels) == name
#     plt.scatter(
#         embedding[idx,0],
#         embedding[idx,1],
#         s=5,
#         alpha=0.6,
#         label=name
#     )
#
# plt.legend()
# plt.title("UMAP of Scheduler Distributions")
# plt.savefig("umap.png")


def compute_mmd(X, Y, gamma=None):

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)

    return XX.mean() + YY.mean() - 2 * XY.mean()


print("\n========== MMD ==========")

pairs = [
    ("RR","WF",rr_scaled,wf_scaled),
    ("RR","PF",rr_scaled,pf_scaled),
    ("WF","PF",wf_scaled,pf_scaled)
]

for a,b,A,B in pairs:

    mmd = compute_mmd(A,B)

    print(f"{a} vs {b}: {mmd:.4f}")


def compare(name1, X1, name2, X2):

    X = np.vstack([X1,X2])

    y = np.hstack([
        np.zeros(len(X1)),
        np.ones(len(X2))
    ])

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        clf,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print(
        f"{name1} vs {name2}: "
        f"{scores.mean():.4f} ± {scores.std():.4f}"
    )


print("\n========== Pairwise Accuracy ==========")

compare("RR",rr_scaled,"WF",wf_scaled)
compare("RR",rr_scaled,"PF",pf_scaled)
compare("WF",wf_scaled,"PF",pf_scaled)