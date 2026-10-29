"""
Unit 2 Lab · Data Preparation & Exploration
Turn three messy SwiftTel source files into one clean, model-ready table, then explore it.

Files  : crm_customers.csv          (CRM export: labels, symbols, two date formats, duplicates, gaps)
         usage_dec2024.csv          (usage system: data GB, voice minutes, recharges; meter glitches)
         support_tickets_q4_2024.csv (one row per complaint ticket, Oct–Dec 2024)
Needs  : pandas, numpy, matplotlib, scipy, scikit-learn
         pip install pandas numpy matplotlib scipy scikit-learn
Run    : python unit2_lab.py   (all three CSV files in the same folder)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
pd.set_option("display.width", 130)

# ============================================================
# STEP 1 · DATA COLLECTION: READ THE THREE SOURCES
# ============================================================
# Read everything as text first (dtype=str) so pandas does not guess wrongly or hide problems.
crm = pd.read_csv(HERE / "crm_customers.csv", dtype=str, keep_default_na=False)
usage = pd.read_csv(HERE / "usage_dec2024.csv")
tickets = pd.read_csv(HERE / "support_tickets_q4_2024.csv", parse_dates=["ticket_date"])
print("CRM:", crm.shape, " usage:", usage.shape, " tickets:", tickets.shape)
print(crm.head(3).to_string())

# ============================================================
# STEP 2 · DATA CLEANING: DUPLICATES, LABELS, TYPES
# ============================================================
print("\nExact duplicate rows:", crm.duplicated().sum())
crm = crm.drop_duplicates()
print("Plan labels before:", crm["plan_type"].value_counts().to_dict())
crm["plan_type"] = (crm["plan_type"].str.lower().str.replace("-", "", regex=False)
                    .map({"prepaid": "Prepaid", "postpaid": "Postpaid"}))
crm["gender"] = crm["gender"].str.strip().str.lower().map({"m": "Male", "male": "Male", "f": "Female", "female": "Female"})
crm["churned"] = crm["churned"].str.title()                        # "yes" -> "Yes"
print("Plan labels after :", crm["plan_type"].value_counts().to_dict())

# "₹399", "399.00", "399" -> 399.0
crm["monthly_charge"] = pd.to_numeric(crm["monthly_charge"].str.replace("₹", "", regex=False))
# "2023-06-29" and "29/06/2023" -> one real date type
crm["join_date"] = pd.to_datetime(crm["join_date"], format="mixed", dayfirst=True)
crm["age"] = pd.to_numeric(crm["age"].replace("", np.nan))
crm["satisfaction"] = pd.to_numeric(crm["satisfaction"].replace("", np.nan))
print(crm.dtypes)

# ============================================================
# STEP 3 · JOIN THE SOURCES INTO ONE TABLE
# ============================================================
usage["customer_id"] = usage["customer_id"].str.upper()           # "st10007" -> "ST10007" before joining!
calls = tickets.groupby("customer_id").size().rename("support_calls")   # 1 row per ticket -> 1 count per customer
df = crm.merge(usage, on="customer_id", how="left").merge(calls, on="customer_id", how="left")
df["support_calls"] = df["support_calls"].fillna(0).astype(int)   # no ticket = 0 calls (a fact, not a gap)
print("\nMerged:", df.shape)
print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])

# ============================================================
# STEP 4 · OUTLIER DETECTION
# ============================================================
def iqr_fences(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)

lo, hi = iqr_fences(df["data_gb"].dropna())
z = (df["data_gb"] - df["data_gb"].mean()) / df["data_gb"].std()
print(f"\ndata_gb IQR fences: [{lo:.2f}, {hi:.2f}]  -> flagged {((df['data_gb'] < lo) | (df['data_gb'] > hi)).sum()}")
print("data_gb |z| > 3      -> flagged", (z.abs() > 3).sum())
print("data_gb > 100 GB (impossible on any plan):", (df["data_gb"] > 100).sum())
print("monthly_charge > 999 (no such plan):", (df["monthly_charge"] > 999).sum(), df.loc[df["monthly_charge"] > 999, "monthly_charge"].tolist())
print("age outside 18-100:", ((df["age"] < 18) | (df["age"] > 100)).sum(), sorted(df.loc[(df["age"] < 18) | (df["age"] > 100), "age"].tolist()))

# treatment: fix what we can explain, blank what is impossible, KEEP real extreme values
df.loc[df["monthly_charge"] > 999, "monthly_charge"] /= 100       # 39900 was a typo for 399
df.loc[df["data_gb"] > 100, "data_gb"] = np.nan                   # meter glitch -> unknown
df.loc[(df["age"] < 18) | (df["age"] > 100), "age"] = np.nan     # impossible -> unknown

# ============================================================
# STEP 5 · HANDLING MISSING VALUES
# ============================================================
print("\nAge missing by plan (%):", (df.groupby("plan_type")["age"].apply(lambda s: s.isna().mean()) * 100).round(1).to_dict())
print("Churn % if satisfaction missing / present:",
      round(df.loc[df["satisfaction"].isna(), "churned"].eq("Yes").mean() * 100, 1),
      round(df.loc[df["satisfaction"].notna(), "churned"].eq("Yes").mean() * 100, 1))
observed = df["age"].dropna()
print(f"Age std: observed {observed.std():.2f}, after mean fill {df['age'].fillna(observed.mean()).std():.2f}")

df["age_was_missing"] = df["age"].isna().astype(int)              # keep the fact that it was missing
df["age"] = df["age"].fillna(df.groupby("plan_type")["age"].transform("median"))
df["satisfaction_missing"] = df["satisfaction"].isna().astype(int)
df["satisfaction"] = df["satisfaction"].fillna(df["satisfaction"].median())
for col in ["data_gb", "voice_minutes", "recharge_count"]:
    df[col] = df[col].fillna(df[col].median())
print("Missing values left:", int(df.isna().sum().sum()))

# ============================================================
# STEP 6 · DATA TRANSFORMATION
# ============================================================
df["tenure_months"] = np.floor((pd.Timestamp("2025-01-01") - df["join_date"]).dt.days / 30.4).astype(int).clip(1, 72)
df["churn_flag"] = (df["churned"] == "Yes").astype(int)
df["charge_minmax"] = (df["monthly_charge"] - df["monthly_charge"].min()) / (df["monthly_charge"].max() - df["monthly_charge"].min())
df["charge_z"] = (df["monthly_charge"] - df["monthly_charge"].mean()) / df["monthly_charge"].std()
df["log_voice"] = np.log1p(df["voice_minutes"])
df["tenure_band"] = pd.cut(df["tenure_months"], [0, 6, 12, 24, 48, 72], labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
df["tier_ordinal"] = df["city_tier"].map({"Tier-1": 1, "Tier-2": 2, "Tier-3": 3})       # ordinal -> ordered numbers
df = pd.concat([df, pd.get_dummies(df["plan_type"], prefix="plan", dtype=int)], axis=1)  # nominal -> one-hot
df["autopay_yes"] = (df["autopay"] == "Yes").astype(int)
df["male"] = (df["gender"] == "Male").astype(int)
print("\nSkew of voice_minutes:", round(df["voice_minutes"].skew(), 2), " after log:", round(df["log_voice"].skew(), 2))
print(df[["customer_id", "monthly_charge", "charge_minmax", "charge_z", "tenure_months", "tenure_band"]].head(3).round(3).to_string(index=False))

# ============================================================
# STEP 7 · FEATURE SELECTION
# ============================================================
features = ["tenure_months", "monthly_charge", "data_gb", "voice_minutes", "support_calls", "age", "satisfaction",
            "recharge_count", "plan_Prepaid", "autopay_yes", "tier_ordinal", "male", "satisfaction_missing"]
r = df[features + ["churn_flag"]].corr()["churn_flag"].drop("churn_flag")
print("\nFilter 1 · correlation with churn (by size):\n", r.reindex(r.abs().sort_values(ascending=False).index).round(3))
print("Redundancy check: r(data_gb, monthly_charge) =", round(df["data_gb"].corr(df["monthly_charge"]), 3))
for col in ["plan_type", "autopay", "city_tier", "gender"]:
    chi2, p, dof, _ = chi2_contingency(pd.crosstab(df[col], df["churned"]), correction=False)
    print(f"Filter 2 · chi-square {col:10} chi2 = {chi2:6.2f}  p = {p:.4f}  {'keep' if p < 0.05 else 'weak'}")
discrete = [c in ("support_calls", "plan_Prepaid", "autopay_yes", "tier_ordinal", "male", "satisfaction_missing", "recharge_count") for c in features]
mi = pd.Series(mutual_info_classif(df[features], df["churn_flag"], discrete_features=discrete, random_state=0), index=features)
print("Filter 3 · mutual information (top 5):", mi.sort_values(ascending=False).head(5).round(4).to_dict())
X = StandardScaler().fit_transform(df[features])
rfe = RFE(LogisticRegression(max_iter=5000), n_features_to_select=5).fit(X, df["churn_flag"])
print("Wrapper · RFE keeps:", [f for f, keep in zip(features, rfe.support_) if keep])

# ============================================================
# STEP 8 · EXPLORATORY DATA ANALYSIS
# ============================================================
print("\nUnivariate summary:\n", df[["age", "tenure_months", "monthly_charge", "data_gb", "voice_minutes", "support_calls"]].describe().round(1).T)
print("\nCustomers aged exactly 18, 19, 20:", [int((df["age"] == a).sum()) for a in (18, 19, 20)], " <- a pile-up worth checking")
print("\nBivariate · churn % by tier and plan:\n", (df.pivot_table(index="city_tier", columns="plan_type", values="churn_flag") * 100).round(1))

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
axes[0, 0].hist(df["voice_minutes"], bins=40); axes[0, 0].set_title("Histogram · voice minutes (right-skewed)")
df.boxplot(column="tenure_months", by="churned", ax=axes[0, 1]); axes[0, 1].set_title("Box plot · tenure by churn"); axes[0, 1].figure.suptitle("")
axes[1, 0].scatter(df["monthly_charge"], df["data_gb"], s=6, alpha=.4); axes[1, 0].set_title("Scatter · charge vs data (r = 0.91)")
cm = df[["tenure_months", "monthly_charge", "data_gb", "support_calls", "satisfaction", "churn_flag"]].corr()
im = axes[1, 1].imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1); axes[1, 1].set_title("Heatmap · correlations")
axes[1, 1].set_xticks(range(len(cm)), cm.columns, rotation=45, ha="right", fontsize=8); axes[1, 1].set_yticks(range(len(cm)), cm.columns, fontsize=8)
fig.colorbar(im, ax=axes[1, 1]); fig.tight_layout()
fig.savefig(HERE / "unit2_eda.png", dpi=110)

df.to_csv(HERE / "swifttel_prepared.csv", index=False)
print("\nSaved swifttel_prepared.csv", df.shape, "and unit2_eda.png")
