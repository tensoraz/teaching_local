"""
Unit 4 Lab · Advanced Modeling & Validation
Ensembles, cross-validation, hyperparameter tuning, evaluation metrics and over/underfitting on SwiftTel churn.

File  : swifttel_customers.csv (from Unit 1)
Needs : pandas, numpy, scikit-learn, matplotlib  ->  pip install pandas numpy scikit-learn matplotlib
Run   : python unit4_lab.py   (a full run takes about a minute: the grid search trains 120 forests)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# ============================================================
# STEP 1 · DATA AND A LOCKED TEST SET
# ============================================================
df = pd.read_csv(HERE / "swifttel_customers.csv")
X = pd.DataFrame({"tenure_months": df["tenure_months"], "monthly_charge": df["monthly_charge"], "support_calls": df["support_calls"],
                  "prepaid": (df["plan_type"] == "Prepaid").astype(int), "autopay": (df["autopay"] == "Yes").astype(int),
                  "tier3": (df["city_tier"] == "Tier-3").astype(int), "satisfaction": df["satisfaction"].astype(float),
                  "age": df["age"], "data_gb": df["data_gb"]})
y = (df["churned"] == "Yes").astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
med = X_train["satisfaction"].median()
X_train, X_test = X_train.fillna({"satisfaction": med}), X_test.fillna({"satisfaction": med})
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
print("train", len(X_train), " test", len(X_test), " churners in test", int(y_test.sum()))

# ============================================================
# STEP 2 · UNDERFITTING AND OVERFITTING
# ============================================================
print("\ndepth  train AUC  CV AUC  test AUC  leaves")
for d in [1, 2, 3, 4, 6, 8, 12, None]:
    tree = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_train, y_train)
    cv = cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=0), X_train, y_train, cv=cv5, scoring="roc_auc").mean()
    print(f"{str(d):>5}  {roc_auc_score(y_train, tree.predict_proba(X_train)[:, 1]):9.3f}  {cv:6.3f}  "
          f"{roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1]):8.3f}  {tree.get_n_leaves():6}")

# ============================================================
# STEP 3 · CROSS-VALIDATION
# ============================================================
logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))   # scaler is refit inside every fold: no leakage
folds = cross_val_score(logit, X_train, y_train, cv=cv5, scoring="roc_auc")
print("\n5-fold AUC:", np.round(folds, 3), " mean", round(folds.mean(), 3), " sd", round(folds.std(), 3))

# ============================================================
# STEP 4 · HYPERPARAMETER TUNING WITH GRID SEARCH
# ============================================================
grid = {"n_estimators": [100, 300], "max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 5, 20]}
search = GridSearchCV(RandomForestClassifier(random_state=0, n_jobs=-1), grid, cv=cv5, scoring="roc_auc").fit(X_train, y_train)
print("\ncombinations:", len(search.cv_results_["params"]), " models trained:", 5 * len(search.cv_results_["params"]))
print("best:", search.best_params_, " CV AUC", round(search.best_score_, 3))

# ============================================================
# STEP 5 · ENSEMBLES VS SINGLE MODELS
# ============================================================
models = {
    "Logistic regression": logit,
    "Decision tree (depth 4)": DecisionTreeClassifier(max_depth=4, random_state=0),
    "Random forest (tuned)": RandomForestClassifier(random_state=0, n_jobs=-1, oob_score=True, **search.best_params_),
    "Gradient boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=0),
    "AdaBoost": AdaBoostClassifier(n_estimators=150, learning_rate=0.5, random_state=0),
}
cv_auc, probs = {}, {}
print(f"\n{'model':24} CV AUC (train)   test AUC")
for name, m in models.items():
    cv_auc[name] = cross_val_score(m, X_train, y_train, cv=cv5, scoring="roc_auc").mean()
    probs[name] = m.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    print(f"{name:24} {cv_auc[name]:.3f}            {roc_auc_score(y_test, probs[name]):.3f}")
print("random forest out-of-bag accuracy:", round(models["Random forest (tuned)"].oob_score_, 3))

# ============================================================
# STEP 6 · EVALUATION METRICS AND THE THRESHOLD
# ============================================================
best = max(cv_auc, key=cv_auc.get)                 # pick by cross-validation, NOT by the test set
p = probs[best]
print("\nchosen by CV:", best)
for thr in (0.5, 0.25):
    tn, fp, fn, tp = confusion_matrix(y_test, (p >= thr).astype(int)).ravel()
    print(f"threshold {thr}: TP {tp} FP {fp} FN {fn} TN {tn} | accuracy {(tp + tn) / len(y_test):.3f} precision {tp / (tp + fp):.3f} "
          f"recall {tp / (tp + fn):.3f} specificity {tn / (tn + fp):.3f} F1 {f1_score(y_test, (p >= thr).astype(int)):.3f}")

# business cost: missing a churner loses ~6 months of revenue (Rs 2,800); a wasted call + voucher costs Rs 300
cost = {round(t, 2): (lambda c: c[2] * 2800 + c[1] * 300)(confusion_matrix(y_test, (p >= t).astype(int)).ravel())
        for t in np.arange(0.05, 0.96, 0.01)}
t_best = min(cost, key=cost.get)
print(f"lowest-cost threshold {t_best}: Rs {cost[t_best]:,}   (at 0.5: Rs {cost[0.5]:,})")

# ============================================================
# STEP 7 · ROC CURVES
# ============================================================
fig, ax = plt.subplots(figsize=(6, 5))
for name, pp in probs.items():
    f, t, _ = roc_curve(y_test, pp)
    ax.plot(f, t, label=f"{name} ({roc_auc_score(y_test, pp):.3f})")
ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(HERE / "unit4_roc.png", dpi=110)
print("\nSaved unit4_roc.png")
