"""Builds BCA 3 Predictive Analytics Unit 4 numbers, charts and the JSON used by the in-browser labs.

Data: SwiftTel churn from Unit 1. Run from the project root:  python tools/build_bca3_unit4_assets.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
UNIT = ROOT / "courses" / "bca3-ds-ai" / "unit-4"
LAB, IMG = UNIT / "lab", UNIT / "img"
LAB.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "courses/bca3-ds-ai/unit-1/lab/swifttel_customers.csv")
X = pd.DataFrame({"tenure_months": df.tenure_months, "monthly_charge": df.monthly_charge, "support_calls": df.support_calls,
                  "prepaid": (df.plan_type == "Prepaid").astype(int), "autopay": (df.autopay == "Yes").astype(int),
                  "tier3": (df.city_tier == "Tier-3").astype(int), "satisfaction": df.satisfaction.astype(float),
                  "age": df.age, "data_gb": df.data_gb})
y = (df.churned == "Yes").astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
med = Xtr.satisfaction.median(); Xtr = Xtr.fillna({"satisfaction": med}); Xte = Xte.fillna({"satisfaction": med})
print("train", len(Xtr), "test", len(Xte), "test churners", int(yte.sum()))

# ---------------- 1. overfitting: tree depth ----------------
depth_rows = []
cv5 = StratifiedKFold(5, shuffle=True, random_state=1)
for d in list(range(1, 16)) + [None]:
    t = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
    cv = cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=0), Xtr, ytr, cv=cv5, scoring="roc_auc").mean()
    depth_rows.append({"depth": d if d else "none", "train_acc": round(t.score(Xtr, ytr), 4), "test_acc": round(t.score(Xte, yte), 4),
                       "train_auc": round(roc_auc_score(ytr, t.predict_proba(Xtr)[:, 1]), 4),
                       "test_auc": round(roc_auc_score(yte, t.predict_proba(Xte)[:, 1]), 4), "cv_auc": round(cv, 4), "leaves": int(t.get_n_leaves())})
for r in depth_rows:
    print("depth", r)

# ---------------- 2. a single split is noisy ----------------
accs, aucs = [], []
for seed in range(50):
    a, b, c, d_ = train_test_split(X.fillna({"satisfaction": 4.0}), y, test_size=0.3, random_state=seed, stratify=y)
    m = LogisticRegression(max_iter=5000).fit(a, c)
    accs.append(m.score(b, d_)); aucs.append(roc_auc_score(d_, m.predict_proba(b)[:, 1]))
print(f"\n50 random splits, logistic: accuracy {min(accs):.3f}-{max(accs):.3f} (mean {np.mean(accs):.3f}, sd {np.std(accs):.3f}); "
      f"AUC {min(aucs):.3f}-{max(aucs):.3f} (mean {np.mean(aucs):.3f}, sd {np.std(aucs):.3f})")

# ---------------- 3. k-fold cross-validation ----------------
logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
fold_auc = cross_val_score(logit, Xtr, ytr, cv=cv5, scoring="roc_auc")
print("5-fold AUC (logistic):", np.round(fold_auc, 4).tolist(), "mean", round(fold_auc.mean(), 4), "sd", round(fold_auc.std(), 4))
fold_sizes = [len(te_i) for _, te_i in cv5.split(Xtr, ytr)]
print("fold sizes", fold_sizes)

# ---------------- 4. hyperparameter tuning ----------------
grid = {"n_estimators": [100, 300], "max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 5, 20]}
gs = GridSearchCV(RandomForestClassifier(random_state=0, n_jobs=-1), grid, cv=cv5, scoring="roc_auc").fit(Xtr, ytr)
print("\ngrid combos", len(gs.cv_results_["params"]), "fits", len(gs.cv_results_["params"]) * 5)
print("best params", gs.best_params_, "best CV AUC", round(gs.best_score_, 4))
cvr = pd.DataFrame(gs.cv_results_)[["param_n_estimators", "param_max_depth", "param_min_samples_leaf", "mean_test_score"]]
print(cvr.sort_values("mean_test_score", ascending=False).head(5).to_string(index=False))
print("worst:", cvr.sort_values("mean_test_score").head(2).to_string(index=False))
kg = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsClassifier()), {"kneighborsclassifier__n_neighbors": [1, 5, 15, 31, 51, 101]},
                  cv=cv5, scoring="roc_auc").fit(Xtr, ytr)
print("KNN k tuning:", dict(zip([1, 5, 15, 31, 51, 101], np.round(kg.cv_results_["mean_test_score"], 4))), "best", kg.best_params_)

# ---------------- 5. ensembles vs single models ----------------
models = {
    "Logistic regression": logit,
    "Decision tree (depth 4)": DecisionTreeClassifier(max_depth=4, random_state=0),
    "Random forest (tuned)": RandomForestClassifier(random_state=0, n_jobs=-1, oob_score=True, **gs.best_params_),
    "Gradient boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=0),
    "AdaBoost": AdaBoostClassifier(n_estimators=150, learning_rate=0.5, random_state=0),
}
scores, res = {}, {}
for name, m in models.items():
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]; pr = (p >= .5).astype(int); scores[name] = p
    cv = cross_val_score(m, Xtr, ytr, cv=cv5, scoring="roc_auc")
    res[name] = dict(cv_auc=cv.mean(), cv_sd=cv.std(), auc=roc_auc_score(yte, p), acc=accuracy_score(yte, pr),
                     prec=precision_score(yte, pr, zero_division=0), rec=recall_score(yte, pr), f1=f1_score(yte, pr))
    print(f"{name:24} CV AUC {cv.mean():.3f}±{cv.std():.3f}  test AUC {res[name]['auc']:.3f}  acc {res[name]['acc']:.3f}  "
          f"prec {res[name]['prec']:.3f}  rec {res[name]['rec']:.3f}  F1 {res[name]['f1']:.3f}")
rf = models["Random forest (tuned)"]
print("RF OOB accuracy", round(rf.oob_score_, 4), " n_estimators", rf.n_estimators)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("RF importances", imp.round(3).to_dict())
gb = models["Gradient boosting"]
stage_auc = [roc_auc_score(yte, p[:, 1]) for p in gb.staged_predict_proba(Xte)]
print("GB staged test AUC at 1,10,50,100,150:", [round(stage_auc[i - 1], 4) for i in (1, 10, 50, 100, 150)])

# ---------------- 6. thresholds, confusion matrix, cost ----------------
best = max(res, key=lambda k: res[k]["cv_auc"])          # choose by cross-validation, never by the test set
p = scores[best]
print("\nbest model by CV AUC:", best)
for thr in (0.5, 0.3, 0.25, 0.2):
    pr = (p >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(yte, pr).ravel()
    print(f"thr {thr}: TP {tp} FP {fp} FN {fn} TN {tn}  acc {(tp + tn) / len(yte):.3f} prec {tp / max(1, tp + fp):.3f} "
          f"rec {tp / (tp + fn):.3f} spec {tn / (tn + fp):.3f} F1 {f1_score(yte, pr):.3f}")
COST_FN, COST_FP = 2800, 300
costs = []
for thr in np.round(np.arange(0.05, 0.96, 0.01), 2):
    pr = (p >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(yte, pr).ravel()
    costs.append((thr, fn * COST_FN + fp * COST_FP, tp, fp, fn, tn))
cbest = min(costs, key=lambda c: c[1])
print(f"cost: FN Rs {COST_FN}, FP Rs {COST_FP}; best threshold {cbest[0]} cost Rs {cbest[1]:,} (TP {cbest[2]} FP {cbest[3]} FN {cbest[4]} TN {cbest[5]})")
print("cost at 0.5:", [c for c in costs if c[0] == 0.5][0][1], " call nobody:", int(yte.sum()) * COST_FN, " call everyone:", int((1 - yte).sum()) * COST_FP)
fpr, tpr, thr_ = roc_curve(yte, p)
pos, neg = p[yte.values == 1], p[yte.values == 0]
print("AUC as ranking probability:", round((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean(), 4))
(LAB / "test_scores.json").write_text(json.dumps({"model": best, "auc": round(res[best]["auc"], 4),
    "rows": [[round(float(a), 4), int(b)] for a, b in zip(p, yte.values)]}), encoding="utf-8")
(LAB / "depth_results.json").write_text(json.dumps(depth_rows), encoding="utf-8")
(LAB / "cv_folds.json").write_text(json.dumps({"auc": np.round(fold_auc, 4).tolist(), "sizes": fold_sizes}), encoding="utf-8")
print("json exports written")

# ---------------- charts ----------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 130})
BLUE, ORANGE, GREY, GREEN, RED, VIOLET = "#2a6fdb", "#e8833a", "#9aa4b2", "#1f9d6b", "#d64545", "#7a5af5"
def save(fig, name):
    fig.tight_layout(); fig.savefig(IMG / name, facecolor="white"); plt.close(fig)

dr = [r for r in depth_rows if r["depth"] != "none"]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.plot([r["depth"] for r in dr], [r["train_auc"] for r in dr], marker="o", color=BLUE, label="training AUC")
ax.plot([r["depth"] for r in dr], [r["cv_auc"] for r in dr], marker="s", color=GREEN, label="5-fold CV AUC")
ax.plot([r["depth"] for r in dr], [r["test_auc"] for r in dr], marker="^", color=ORANGE, label="test AUC")
bd = max(dr, key=lambda r: r["cv_auc"])
ax.axvline(bd["depth"], color=GREY, ls="--", lw=1); ax.text(bd["depth"] + .2, .55, f"best CV\ndepth {bd['depth']}", fontsize=9, color="#555")
ax.text(1.1, .52, "underfitting", fontsize=9, color=RED); ax.text(10, .52, "overfitting", fontsize=9, color=RED)
ax.set_xlabel("tree max_depth"); ax.set_ylabel("AUC"); ax.set_ylim(.5, 1.01); ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("Deeper trees: training keeps improving, test does not")
save(fig, "overfit_curve.png")

fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.3))
axes[0].hist(accs, bins=12, color=BLUE); axes[0].set_title("Test accuracy, 50 random splits"); axes[0].set_xlabel("accuracy")
axes[1].bar([f"fold {i + 1}" for i in range(5)], fold_auc, color=GREEN); axes[1].axhline(fold_auc.mean(), color=ORANGE, ls="--")
axes[1].set_ylim(.6, .8); axes[1].set_title(f"5-fold CV AUC: {fold_auc.mean():.3f} ± {fold_auc.std():.3f}")
for i, v in enumerate(fold_auc):
    axes[1].text(i, v + .004, f"{v:.3f}", ha="center", fontsize=9)
save(fig, "cv_variation.png")

fig, ax = plt.subplots(figsize=(6, 5))
cols = {"Logistic regression": BLUE, "Decision tree (depth 4)": GREY, "Random forest (tuned)": GREEN, "Gradient boosting": ORANGE, "AdaBoost": VIOLET}
for name, pp in scores.items():
    f, t, _ = roc_curve(yte, pp); ax.plot(f, t, color=cols[name], lw=2, label=f"{name} (AUC {res[name]['auc']:.3f})")
ax.plot([0, 1], [0, 1], color="#bbb", ls="--", label="random guess (0.5)")
ax.set_xlabel("false positive rate (1 − specificity)"); ax.set_ylabel("true positive rate (recall)"); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
ax.set_title("ROC curves on 450 test customers")
save(fig, "roc_curves.png")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, thr in zip(axes, (0.5, cbest[0])):
    cm = confusion_matrix(yte, (p >= thr).astype(int))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{['TN', 'FP'][j] if i == 0 else ['FN', 'TP'][j]}\n{cm[i, j]}", ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred: stay", "pred: churn"]); ax.set_yticks([0, 1], ["actual: stay", "actual: churn"])
    ax.set_title(f"threshold {thr}")
save(fig, "confusion_matrices.png")

ths = np.round(np.arange(0.05, 0.91, 0.01), 2)
P, R, F = [], [], []
for t_ in ths:
    pr = (p >= t_).astype(int); P.append(precision_score(yte, pr, zero_division=0)); R.append(recall_score(yte, pr)); F.append(f1_score(yte, pr))
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(ths, P, color=BLUE, label="precision"); ax.plot(ths, R, color=ORANGE, label="recall"); ax.plot(ths, F, color=GREEN, label="F1")
ax.axvline(.5, color=GREY, ls=":", lw=1); ax.axvline(cbest[0], color=RED, ls="--", lw=1)
ax.text(cbest[0] + .01, .92, "lowest cost", color=RED, fontsize=9)
ax.set_xlabel("decision threshold"); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=9); ax.set_title("Precision–recall trade-off")
save(fig, "threshold_tradeoff.png")
best_f1 = ths[int(np.argmax(F))]; print("best F1 threshold", best_f1, "F1", round(max(F), 3))

fig, ax = plt.subplots(figsize=(6.4, 3.4))
imp.sort_values().plot.barh(ax=ax, color=[GREEN if v >= imp.iloc[2] else GREY for v in imp.sort_values()])
ax.set_title("Random forest feature importance"); ax.set_xlabel("mean decrease in impurity")
save(fig, "rf_importance.png")
print("charts written")
