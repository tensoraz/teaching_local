"""Builds BCA 3 Predictive Analytics Unit 5 datasets, charts, a saved model and the JSON used by the in-browser labs.

Finance case : credit_applications.csv (generated, fixed seed). Rural applicants have "thin" credit files, so their
               observed credit score UNDERSTATES their true creditworthiness: a realistic proxy-bias problem.
Healthcare   : clinic_screening.csv (generated)
Marketing    : K-means on Unit 2's prepared SwiftTel table
Run from the project root after Units 1-2:  python tools/build_bca3_unit5_assets.py
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, silhouette_score, recall_score, precision_score, confusion_matrix

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
UNIT = ROOT / "courses" / "bca3-ds-ai" / "unit-5"
LAB, IMG = UNIT / "lab", UNIT / "img"
LAB.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(505)

# ============================== finance: credit applications ==============================
N = 2000
region = rng.choice(["Urban", "Rural"], N, p=[0.62, 0.38])
gender = rng.choice(["Male", "Female"], N, p=[0.58, 0.42])
age = np.clip(rng.normal(36, 9, N), 21, 65).astype(int)
emp = rng.choice(["Salaried", "Self-employed"], N, p=[0.64, 0.36])
years = np.clip(np.round(rng.gamma(2, 3, N)), 0, 35).astype(int)
income = np.round(np.exp(rng.normal(np.where(region == "Urban", np.log(6.0), np.log(3.6)), 0.45, N)), 2)   # Rs lakh / year
loan = np.round(income * rng.uniform(0.4, 2.6, N), 2)
emi_ratio = np.round(np.clip(loan * 0.26 / income, 0.05, 0.80), 3)                                          # share of income on EMIs
past = rng.choice([0, 1, 2], N, p=[0.82, 0.14, 0.04])
true_score = np.clip(np.round(rng.normal(715, 55, N) - 45 * past), 300, 900)
thin = np.where(region == "Rural", rng.normal(40, 15, N), rng.normal(0, 8, N))                              # thin files understate rural scores
credit_score = np.clip(np.round(true_score - thin), 300, 900).astype(int)
z = (-2.25 + 4.0 * (emi_ratio - 0.3) - 0.013 * (true_score - 715) + 0.85 * past + 0.35 * (emp == "Self-employed")
     - 0.04 * (years - 5) - 0.06 * (income - 5))
default = (rng.random(N) < 1 / (1 + np.exp(-z))).astype(int)
credit = pd.DataFrame({"application_id": [f"LN{70001 + i}" for i in range(N)], "age": age, "gender": gender, "region": region,
                       "employment": emp, "years_employed": years, "income_lakh": income, "loan_amount_lakh": loan,
                       "emi_to_income": emi_ratio, "credit_score": credit_score, "past_defaults": past, "defaulted": default})
credit.to_csv(LAB / "credit_applications.csv", index=False)
print("credit", credit.shape, "default rate", round(default.mean(), 4), " by region", credit.groupby("region").defaulted.mean().round(4).to_dict(),
      " by gender", credit.groupby("gender").defaulted.mean().round(4).to_dict())
print("mean credit score by region", credit.groupby("region").credit_score.mean().round(1).to_dict())

FEATS = ["age", "years_employed", "income_lakh", "loan_amount_lakh", "emi_to_income", "credit_score", "past_defaults", "employment"]
NUM = FEATS[:-1]
tr, te = train_test_split(credit, test_size=0.3, random_state=42, stratify=credit.defaulted)
pipe = Pipeline([("prep", ColumnTransformer([("num", StandardScaler(), NUM),
                                             ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["employment"])])),
                 ("model", LogisticRegression(max_iter=5000))]).fit(tr[FEATS], tr.defaulted)
p_te = pipe.predict_proba(te[FEATS])[:, 1]
print("credit logistic test AUC", round(roc_auc_score(te.defaulted, p_te), 4))
joblib.dump(pipe, LAB / "credit_model.joblib")

# coefficients on the original scale (for reason codes in the browser)
sc = pipe.named_steps["prep"].named_transformers_["num"]
b = pipe.named_steps["model"].coef_[0]; b0 = pipe.named_steps["model"].intercept_[0]
names = NUM + ["self_employed"]
raw = {n: float(b[i] / sc.scale_[i]) for i, n in enumerate(NUM)}; raw["self_employed"] = float(b[-1])
intercept_raw = float(b0 - sum(b[i] * sc.mean_[i] / sc.scale_[i] for i in range(len(NUM))))
means = {n: float(sc.mean_[i]) for i, n in enumerate(NUM)}; means["self_employed"] = float((tr.employment == "Self-employed").mean())
print("standardised coefs", dict(zip(names, np.round(b, 3))), "intercept", round(b0, 3))
print("raw coefs", {k: round(v, 5) for k, v in raw.items()}, "raw intercept", round(intercept_raw, 4))
print("odds ratio per +50 credit points", round(float(np.exp(raw['credit_score'] * 50)), 3), " per +0.1 EMI ratio", round(float(np.exp(raw['emi_to_income'] * 0.1)), 3),
      " per past default", round(float(np.exp(raw['past_defaults'])), 3))

# permutation importance vs coefficient size
perm = permutation_importance(pipe, te[FEATS], te.defaulted, scoring="roc_auc", n_repeats=20, random_state=0)
pi = pd.Series(perm.importances_mean, index=FEATS).sort_values(ascending=False)
print("permutation importance (AUC drop)", pi.round(4).to_dict())

# a declined applicant and reason codes: contribution = w * (x - mean)
THR = 0.25
te_scored = te.assign(p_default=p_te)
ex = te_scored[(te_scored.p_default > THR)].sort_values("p_default").iloc[len(te_scored[te_scored.p_default > THR]) // 2]
xv = {n: float(ex[n]) for n in NUM}; xv["self_employed"] = float(ex.employment == "Self-employed")
contrib = {n: raw[n] * (xv[n] - means[n]) for n in names}
zz = intercept_raw + sum(raw[n] * xv[n] for n in names)
print("example applicant", ex[FEATS + ["region", "defaulted"]].to_dict(), "p", round(float(ex.p_default), 4), "check", round(1 / (1 + np.exp(-zz)), 4))
print("contributions", {k: round(v, 3) for k, v in sorted(contrib.items(), key=lambda kv: -kv[1])})
base_z = intercept_raw + sum(raw[n] * means[n] for n in names)
print("baseline (average applicant) p", round(1 / (1 + np.exp(-base_z)), 4))
need = (np.log(THR / (1 - THR)) - (zz - raw["credit_score"] * xv["credit_score"])) / raw["credit_score"]
print("credit score needed to fall below threshold", round(need, 1))
(LAB / "credit_model.json").write_text(json.dumps({"intercept": intercept_raw, "coef": raw, "mean": means, "threshold": THR,
                                                    "example": {**{n: xv[n] for n in names}}}), encoding="utf-8")

# partial dependence of P(default) on credit score (average over test applicants)
grid = np.arange(500, 851, 25)
pdp = []
for g in grid:
    tmp = te[FEATS].copy(); tmp["credit_score"] = g; pdp.append(pipe.predict_proba(tmp)[:, 1].mean())
print("PDP credit score", dict(zip(grid.tolist(), np.round(pdp, 4).tolist())))

# fairness audit at the threshold
te_scored["approved"] = (te_scored.p_default < THR).astype(int)
fair = {}
for col in ["region", "gender"]:
    for g, d in te_scored.groupby(col):
        good = d[d.defaulted == 0]
        fair[f"{col}={g}"] = dict(n=len(d), approve=round(d.approved.mean(), 4), tpr_good=round(good.approved.mean(), 4),
                                  default_among_approved=round(d[d.approved == 1].defaulted.mean(), 4), actual_default=round(d.defaulted.mean(), 4),
                                  mean_p=round(d.p_default.mean(), 4))
print("fairness", json.dumps(fair, indent=0))
for col, (a, b_) in {"region": ("Rural", "Urban"), "gender": ("Female", "Male")}.items():
    print(f"disparate impact {col}: {fair[f'{col}={a}']['approve'] / fair[f'{col}={b_}']['approve']:.3f}")
(LAB / "fairness_scores.json").write_text(json.dumps([[round(float(p_), 4), int(y_), r_, g_] for p_, y_, r_, g_ in
    zip(te_scored.p_default, te_scored.defaulted, te_scored.region, te_scored.gender)]), encoding="utf-8")

# drift: credit score distribution, training vs a "new month" where a partner bank sends more thin-file applicants
new_month = np.clip(np.round(rng.normal(652, 70, 600)), 300, 900)
edges = np.quantile(tr.credit_score, np.linspace(0, 1, 11)); edges[0], edges[-1] = 300, 901
e_ = np.histogram(tr.credit_score, edges)[0] / len(tr); a_ = np.histogram(new_month, edges)[0] / len(new_month)
psi = float(np.sum((a_ - e_) * np.log(a_ / e_)))
print("PSI bins expected", np.round(e_, 3).tolist(), "actual", np.round(a_, 3).tolist(), "PSI", round(psi, 4),
      " mean train", round(tr.credit_score.mean(), 1), " new", round(new_month.mean(), 1))
pd.DataFrame({"credit_score": new_month.astype(int)}).to_csv(LAB / "credit_scores_this_month.csv", index=False)
(LAB / "drift_data.json").write_text(json.dumps({"train": tr.credit_score.astype(int).tolist(), "edges": edges.tolist()}), encoding="utf-8")

# batch scoring demo: 20 new applications
batch = credit.sample(20, random_state=9)[["application_id"] + FEATS]
batch.to_csv(LAB / "new_applications.csv", index=False)

# ============================== healthcare: diabetes screening ==============================
M = 1500
h_age = np.clip(rng.normal(45, 13, M), 18, 85).astype(int)
bmi = np.round(np.clip(rng.normal(26, 4.5, M), 15, 48), 1)
glucose = np.round(np.clip(rng.normal(102, 18, M) + 0.35 * (bmi - 26) * 3, 60, 260)).astype(int)
bp = np.round(np.clip(rng.normal(126, 15, M) + 0.4 * (h_age - 45), 90, 200)).astype(int)
fam = rng.choice([0, 1], M, p=[0.7, 0.3])
act = rng.choice(["Low", "Medium", "High"], M, p=[0.4, 0.4, 0.2])
hz = -2.0 + 0.045 * (glucose - 100) + 0.09 * (bmi - 26) + 0.03 * (h_age - 45) + 0.8 * fam + 0.5 * (act == "Low") - 0.4 * (act == "High")
diab = (rng.random(M) < 1 / (1 + np.exp(-hz))).astype(int)
clinic = pd.DataFrame({"patient_id": [f"PT{90001 + i}" for i in range(M)], "age": h_age, "bmi": bmi, "fasting_glucose": glucose,
                       "systolic_bp": bp, "family_history": fam, "activity_level": act, "diabetic": diab})
clinic.to_csv(LAB / "clinic_screening.csv", index=False)
hX = clinic[["age", "bmi", "fasting_glucose", "systolic_bp", "family_history"]].assign(low_activity=(clinic.activity_level == "Low").astype(int))
htr, hte, hytr, hyte = train_test_split(hX, clinic.diabetic, test_size=0.3, random_state=42, stratify=clinic.diabetic)
hm = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=5000))]).fit(htr, hytr)
hp = hm.predict_proba(hte)[:, 1]
print("\nclinic", clinic.shape, "diabetic rate", round(diab.mean(), 4), "test AUC", round(roc_auc_score(hyte, hp), 4))
for t in (0.5, 0.3, 0.2, 0.15, 0.1):
    pr = (hp >= t).astype(int); tn, fp, fn, tp = confusion_matrix(hyte, pr).ravel()
    print(f"  thr {t}: recall {tp / (tp + fn):.3f} precision {tp / max(1, tp + fp):.3f} referred {tp + fp} of {len(hyte)}  missed {fn}")

# ============================== marketing: K-means segmentation ==============================
sw = pd.read_csv(ROOT / "courses/bca3-ds-ai/unit-2/lab/swifttel_prepared_reference.csv")
segf = ["data_gb", "log_voice", "monthly_charge", "tenure_months"]
Z = StandardScaler().fit_transform(sw[segf])
inertia = {k: round(float(KMeans(k, n_init=10, random_state=0).fit(Z).inertia_), 1) for k in range(1, 9)}
sil = {k: round(float(silhouette_score(Z, KMeans(k, n_init=10, random_state=0).fit_predict(Z))), 4) for k in range(2, 9)}
print("\nelbow inertia", inertia); print("silhouette", sil)
km = KMeans(4, n_init=10, random_state=0).fit(Z)
sw["segment"] = km.labels_
prof = sw.groupby("segment").agg(customers=("customer_id", "size"), data_gb=("data_gb", "mean"), voice_min=("voice_minutes", "median"),
                                 charge=("monthly_charge", "mean"), tenure=("tenure_months", "mean"), churn=("churn_flag", "mean"),
                                 prepaid=("prepaid", "mean")).round(2)
print(prof.to_string())
sw[["customer_id", "segment"]].to_csv(LAB / "swifttel_segments.csv", index=False)
samp = sw.sample(600, random_state=3)
(LAB / "kmeans_points.json").write_text(json.dumps({"x": samp.data_gb.round(1).tolist(), "y": samp.voice_minutes.astype(int).tolist(),
                                                     "churn": samp.churn_flag.astype(int).tolist()}), encoding="utf-8")
sw.to_csv(LAB / "swifttel_prepared.csv", index=False, columns=["customer_id", "data_gb", "voice_minutes", "log_voice", "monthly_charge",
                                                             "tenure_months", "prepaid", "churn_flag"])

# ============================== charts ==============================
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 130})
BLUE, ORANGE, GREY, GREEN, RED, VIOLET, TEAL = "#2a6fdb", "#e8833a", "#9aa4b2", "#1f9d6b", "#d64545", "#7a5af5", "#149c9c"
def save(fig, name):
    fig.tight_layout(); fig.savefig(IMG / name, facecolor="white"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.6, 3.5))
pi.sort_values().plot.barh(ax=ax, color=[GREEN if v > .02 else GREY for v in pi.sort_values()], xerr=pd.Series(perm.importances_std, index=FEATS)[pi.sort_values().index])
ax.set_xlabel("drop in test AUC when the feature is shuffled"); ax.set_title("Permutation importance: credit model")
save(fig, "perm_importance.png")

fig, ax = plt.subplots(figsize=(6.4, 3.3))
ax.plot(grid, pdp, marker="o", color=BLUE, lw=2); ax.axhline(THR, color=RED, ls="--", lw=1)
ax.text(505, THR + .01, f"approval threshold {THR}", color=RED, fontsize=9)
ax.set_xlabel("credit score (all other details unchanged)"); ax.set_ylabel("average P(default)")
ax.set_title("Partial dependence on credit score"); save(fig, "pdp_credit_score.png")

fig, ax = plt.subplots(figsize=(7, 3.3))
xs = np.arange(10)
ax.bar(xs - .2, e_ * 100, .4, color=GREY, label="training data"); ax.bar(xs + .2, a_ * 100, .4, color=ORANGE, label="new month")
ax.set_xticks(xs, [f"{int(edges[i])}–{int(edges[i + 1]) - 1}" for i in range(10)], rotation=40, ha="right", fontsize=8.5)
ax.set_ylabel("% of applicants"); ax.legend(frameon=False); ax.set_title(f"Credit-score drift: PSI = {psi:.2f}")
save(fig, "psi_drift.png")

fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
axes[0].plot(list(inertia), list(inertia.values()), marker="o", color=BLUE); axes[0].axvline(4, color=GREY, ls="--")
axes[0].set_xlabel("k (number of segments)"); axes[0].set_ylabel("inertia (within-cluster SS)"); axes[0].set_title("Elbow method")
cols = [BLUE, ORANGE, GREEN, VIOLET]
for s_ in range(4):
    d = sw[sw.segment == s_]
    axes[1].scatter(d.data_gb, d.voice_minutes, s=6, alpha=.5, color=cols[s_], label=f"segment {s_} (churn {d.churn_flag.mean():.0%})")
axes[1].set_xlabel("data GB / month"); axes[1].set_ylabel("voice minutes"); axes[1].legend(frameon=False, fontsize=8, markerscale=2)
axes[1].set_title("K-means (k = 4) on 4 usage features")
save(fig, "kmeans_segments.png")

fig, ax = plt.subplots(figsize=(7, 3.3))
groups = ["region=Urban", "region=Rural", "gender=Male", "gender=Female"]
xx = np.arange(4)
ax.bar(xx - .2, [fair[g]["approve"] * 100 for g in groups], .4, color=BLUE, label="approval rate (all applicants)")
ax.bar(xx + .2, [fair[g]["tpr_good"] * 100 for g in groups], .4, color=GREEN, label="approval rate among good payers")
for i, g in enumerate(groups):
    ax.text(i - .2, fair[g]["approve"] * 100 + 1.5, f"{fair[g]['approve'] * 100:.0f}", ha="center", fontsize=9)
    ax.text(i + .2, fair[g]["tpr_good"] * 100 + 1.5, f"{fair[g]['tpr_good'] * 100:.0f}", ha="center", fontsize=9)
ax.set_xticks(xx, ["Urban", "Rural", "Male", "Female"]); ax.set_ylim(0, 118); ax.set_ylabel("%"); ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2)
ax.set_title("Fairness audit at threshold 0.25"); save(fig, "fairness_audit.png")
print("charts written")
