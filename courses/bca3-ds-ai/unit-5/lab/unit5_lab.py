"""
Unit 5 Lab · Model Deployment & Applications
Interpret, save, batch-score and monitor a credit model; audit it for fairness; then two more case studies.

Files : credit_applications.csv  (finance case: 2,000 loan applications)
        new_applications.csv     (20 new applications to score)
        credit_scores_this_month.csv (600 credit scores from this month, for drift monitoring)
        clinic_screening.csv     (healthcare case: diabetes screening)
        swifttel_prepared.csv    (marketing case: customer segmentation, from Unit 2)
Needs : pandas, numpy, scikit-learn, joblib, matplotlib  ->  pip install pandas numpy scikit-learn joblib matplotlib
Run   : python unit5_lab.py   (then try the real-time API: python score_api.py)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, confusion_matrix, silhouette_score

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
pd.set_option("display.width", 130)

# ============================================================
# STEP 1 · FINANCE CASE: TRAIN A PIPELINE (PREPROCESSING + MODEL TOGETHER)
# ============================================================
credit = pd.read_csv(HERE / "credit_applications.csv")
FEATS = ["age", "years_employed", "income_lakh", "loan_amount_lakh", "emi_to_income", "credit_score", "past_defaults", "employment"]
NUM = FEATS[:-1]
train, test = train_test_split(credit, test_size=0.3, random_state=42, stratify=credit["defaulted"])
pipe = Pipeline([
    ("prep", ColumnTransformer([("num", StandardScaler(), NUM),
                                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["employment"])])),
    ("model", LogisticRegression(max_iter=5000)),
]).fit(train[FEATS], train["defaulted"])
p = pipe.predict_proba(test[FEATS])[:, 1]
print("default rate", round(credit["defaulted"].mean(), 3), "  test AUC", round(roc_auc_score(test["defaulted"], p), 3))

# ============================================================
# STEP 2 · INTERPRET THE MODEL (GLOBAL)
# ============================================================
coef = pd.Series(pipe.named_steps["model"].coef_[0], index=NUM + ["self_employed"])
print("\nStandardised coefficients (effect of +1 standard deviation on the log-odds):")
print(coef.reindex(coef.abs().sort_values(ascending=False).index).round(3).to_string())
perm = permutation_importance(pipe, test[FEATS], test["defaulted"], scoring="roc_auc", n_repeats=20, random_state=0)
print("\nPermutation importance (AUC lost when the column is shuffled):")
print(pd.Series(perm.importances_mean, index=FEATS).sort_values(ascending=False).round(4).to_string())
print("\nPartial dependence: average P(default) if every applicant had this credit score")
for s in (550, 650, 700, 750, 800):
    tmp = test[FEATS].copy(); tmp["credit_score"] = s
    print(f"  credit score {s}: {pipe.predict_proba(tmp)[:, 1].mean():.3f}")

# ============================================================
# STEP 3 · EXPLAIN ONE DECISION (LOCAL): REASON CODES
# ============================================================
THRESHOLD = 0.25                                         # approve if P(default) < 0.25
sc = pipe.named_steps["prep"].named_transformers_["num"]
w = pipe.named_steps["model"].coef_[0]
declined = test.assign(p=p)[p > THRESHOLD].sort_values("p")
applicant = declined.iloc[len(declined) // 2]                        # a typical declined applicant
z = (applicant[NUM].astype(float).values - sc.mean_) / sc.scale_   # standardised values
contrib = pd.Series(np.append(w[:-1] * z, w[-1] * (float(applicant["employment"] == "Self-employed") -
                    (train["employment"] == "Self-employed").mean())), index=NUM + ["self_employed"])
print(f"\nApplicant {applicant['application_id']}: P(default) = {applicant['p']:.3f} -> {'DECLINE' if applicant['p'] >= THRESHOLD else 'APPROVE'}")
print("Reasons (contribution to log-odds vs an average applicant; + pushes towards decline):")
print(contrib.sort_values(ascending=False).round(3).head(4).to_string())

# ============================================================
# STEP 4 · DEPLOY: SAVE THE PIPELINE AND BATCH-SCORE NEW APPLICATIONS
# ============================================================
joblib.dump(pipe, HERE / "credit_model_lab.joblib")
loaded = joblib.load(HERE / "credit_model_lab.joblib")      # what a server or nightly job would do
new = pd.read_csv(HERE / "new_applications.csv")
new["p_default"] = loaded.predict_proba(new[FEATS])[:, 1].round(3)
new["decision"] = np.where(new["p_default"] < THRESHOLD, "approve", "refer to officer")
new[["application_id", "p_default", "decision"]].to_csv(HERE / "scored_applications.csv", index=False)
print("\nBatch-scored", len(new), "applications:", new["decision"].value_counts().to_dict())
print(new[["application_id", "credit_score", "emi_to_income", "p_default", "decision"]].head(5).to_string(index=False))

# ============================================================
# STEP 5 · MONITOR: POPULATION STABILITY INDEX (PSI)
# ============================================================
this_month = pd.read_csv(HERE / "credit_scores_this_month.csv")["credit_score"]   # a partner bank sends more thin-file applicants
edges = np.quantile(train["credit_score"], np.linspace(0, 1, 11)); edges[0], edges[-1] = 300, 901
expected = np.histogram(train["credit_score"], edges)[0] / len(train)
actual = np.histogram(this_month, edges)[0] / len(this_month)
psi = np.sum((actual - expected) * np.log(actual / expected))
print(f"\nPSI credit_score = {psi:.3f}  ->  {'no change' if psi < 0.1 else 'watch' if psi < 0.25 else 'SIGNIFICANT DRIFT: investigate / retrain'}")

# ============================================================
# STEP 6 · ETHICS: FAIRNESS AUDIT
# ============================================================
audit = test.assign(p=p, approved=(p < THRESHOLD).astype(int))
for col in ["region", "gender"]:
    rows = []
    for g, d in audit.groupby(col):
        rows.append({col: g, "applicants": len(d), "approval_rate": d["approved"].mean(), "approval_rate_good_payers":
                     d.loc[d["defaulted"] == 0, "approved"].mean(), "actual_default_rate": d["defaulted"].mean()})
    t = pd.DataFrame(rows).set_index(col)
    print("\n" + t.round(3).to_string())
    print(f"disparate impact ratio = {t['approval_rate'].min() / t['approval_rate'].max():.3f}  (below 0.8 is a warning sign)")

# ============================================================
# STEP 7 · HEALTHCARE CASE: SCREENING NEEDS HIGH RECALL
# ============================================================
clinic = pd.read_csv(HERE / "clinic_screening.csv")
hX = clinic[["age", "bmi", "fasting_glucose", "systolic_bp", "family_history"]].assign(low_activity=(clinic["activity_level"] == "Low").astype(int))
a, b, c, d = train_test_split(hX, clinic["diabetic"], test_size=0.3, random_state=42, stratify=clinic["diabetic"])
hm = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=5000))]).fit(a, c)
hp = hm.predict_proba(b)[:, 1]
print(f"\nDiabetes screening: {clinic['diabetic'].mean():.1%} diabetic, test AUC {roc_auc_score(d, hp):.3f}")
for t in (0.5, 0.2, 0.1):
    tn, fp, fn, tp = confusion_matrix(d, (hp >= t).astype(int)).ravel()
    print(f"  threshold {t}: recall {tp / (tp + fn):.3f}  referred for a lab test {tp + fp} of {len(d)}  missed {fn}")

# ============================================================
# STEP 8 · MARKETING CASE: K-MEANS CUSTOMER SEGMENTS
# ============================================================
sw = pd.read_csv(HERE / "swifttel_prepared.csv")
cols = ["data_gb", "log_voice", "monthly_charge", "tenure_months"]
Z = StandardScaler().fit_transform(sw[cols])                         # K-means uses distances: scale first
for k in range(2, 7):
    km = KMeans(k, n_init=10, random_state=0).fit(Z)
    print(f"k={k}: inertia {km.inertia_:7.1f}  silhouette {silhouette_score(Z, km.labels_):.3f}")
sw["segment"] = KMeans(4, n_init=10, random_state=0).fit_predict(Z)
print(sw.groupby("segment").agg(customers=("customer_id", "size"), data_gb=("data_gb", "mean"), voice=("voice_minutes", "median"),
                                charge=("monthly_charge", "mean"), tenure=("tenure_months", "mean"), prepaid=("prepaid", "mean"),
                                churn=("churn_flag", "mean")).round(2).to_string())

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.scatter(sw["data_gb"], sw["voice_minutes"], c=sw["segment"], s=6, cmap="tab10")
ax.set_xlabel("data GB"); ax.set_ylabel("voice minutes"); ax.set_title("SwiftTel segments (K-means, k = 4)")
fig.tight_layout(); fig.savefig(HERE / "unit5_segments.png", dpi=110)
print("\nSaved credit_model_lab.joblib, scored_applications.csv and unit5_segments.png")
