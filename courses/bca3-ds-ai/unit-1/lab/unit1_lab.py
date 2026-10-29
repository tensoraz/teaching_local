"""
Unit 1 Lab · Introduction to Predictive Analytics
One small project through the predictive analytics lifecycle: descriptive → predictive → prescriptive.

Dataset : swifttel_customers.csv  (1,500 customers of a telecom company on 1 Jan 2025;
          churned = Yes if they left during Jan–Mar 2025)
          swifttel_monthly.csv    (company revenue, Jan 2023 – Dec 2024)
Needs   : pandas, numpy, scikit-learn, matplotlib  ->  pip install pandas numpy scikit-learn matplotlib
Run     : python unit1_lab.py     (keep both CSV files in the same folder)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()  # works in Jupyter too
pd.set_option("display.width", 120)

# ============================================================
# STAGE 1 · DEFINE THE BUSINESS PROBLEM
# ============================================================
# SwiftTel loses customers every month ("churn"). The retention team can call
# customers and offer a Rs 300 voucher, but cannot call everyone.
#   Descriptive : how many customers left, and which kinds?
#   Predictive  : which current customers are most likely to leave next?
#   Prescriptive: who should get the Rs 300 offer so we gain the most?

# ============================================================
# STAGE 2 · COLLECT THE DATA AND KNOW ITS TYPES
# ============================================================
df = pd.read_csv(HERE / "swifttel_customers.csv", parse_dates=["join_date"])
print("Shape (rows, columns):", df.shape)
print(df.head(3).to_string())
print("\nPython data types:\n", df.dtypes)

# Measurement level of each column (this is what SPSS calls Measure)
levels = {
    "customer_id": "Nominal (identifier)", "gender": "Nominal", "city_tier": "Ordinal",
    "plan_type": "Nominal", "autopay": "Nominal", "satisfaction": "Ordinal (1-5)",
    "churned": "Nominal (target)", "join_date": "Interval (date)", "age": "Ratio",
    "tenure_months": "Ratio", "monthly_charge": "Ratio", "data_gb": "Ratio", "support_calls": "Ratio (count)",
}
print("\nMeasurement levels:")
for col, lvl in levels.items():
    print(f"  {col:15} {lvl}")

# ============================================================
# STAGE 3 · PREPARE THE DATA
# ============================================================
print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])
df["churn_flag"] = (df["churned"] == "Yes").astype(int)          # Yes/No -> 1/0 for maths
# Missing satisfaction is filled later with the median of the TRAINING data only

# ============================================================
# STAGE 4 · EXPLORE: DESCRIPTIVE AND DIAGNOSTIC ANALYTICS
# ============================================================
print(f"\nOverall churn rate: {df['churn_flag'].mean():.1%}  ({df['churn_flag'].sum()} of {len(df)} customers)")
for col in ["plan_type", "autopay", "city_tier"]:
    print(f"\nChurn rate by {col}:\n", (df.groupby(col)["churn_flag"].mean() * 100).round(1))
calls_band = pd.cut(df["support_calls"], [-1, 0, 1, 2, 3, 9], labels=["0", "1", "2", "3", "4+"])
print("\nChurn rate by support calls:\n", (df.groupby(calls_band, observed=True)["churn_flag"].mean() * 100).round(1))

# diagnostic: how are the leavers different from the stayers?
print("\nAverage values, stayers vs leavers:\n",
      df.groupby("churned")[["tenure_months", "support_calls", "monthly_charge", "satisfaction"]].mean().round(2))

# ============================================================
# STAGE 5 · BUILD A PREDICTIVE MODEL
# ============================================================
# Logistic regression turns customer details into a churn probability (Unit 3 explains how).
X = pd.DataFrame({
    "tenure_months": df["tenure_months"], "monthly_charge": df["monthly_charge"],
    "support_calls": df["support_calls"], "prepaid": (df["plan_type"] == "Prepaid").astype(int),
    "autopay": (df["autopay"] == "Yes").astype(int), "tier3": (df["city_tier"] == "Tier-3").astype(int),
    "satisfaction": df["satisfaction"].astype(float),
})
y = df["churn_flag"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
median_sat = X_train["satisfaction"].median()
X_train = X_train.fillna({"satisfaction": median_sat})
X_test = X_test.fillna({"satisfaction": median_sat})
model = LogisticRegression(max_iter=5000).fit(X_train, y_train)
print("\nLearned weights (+ means more likely to churn):")
for name, w in zip(X.columns, model.coef_[0]):
    print(f"  {name:15} {w:+.4f}")

# ============================================================
# STAGE 6 · EVALUATE THE MODEL
# ============================================================
accuracy = (model.predict(X_test) == y_test).mean()
baseline = 1 - y_test.mean()                       # a "model" that always says "No"
p_test = model.predict_proba(X_test)[:, 1]
top10 = y_test.values[np.argsort(-p_test)[: len(y_test) // 10]]
print(f"\nTest accuracy             : {accuracy:.1%}")
print(f"Always-say-No accuracy    : {baseline:.1%}   <- accuracy alone can fool you!")
print(f"Churn rate in top 10% risk: {top10.mean():.1%}  vs {y_test.mean():.1%} overall"
      f"  -> lift {top10.mean() / y_test.mean():.2f}x")

# ============================================================
# STAGE 7 · DEPLOY: PRESCRIPTIVE DECISION AND A FORECAST
# ============================================================
# Score every customer, then offer Rs 300 only where the expected gain beats the cost.
# Assumption: 40% of would-be leavers accept the offer and stay 6 more months.
df["churn_prob"] = model.predict_proba(X.fillna({"satisfaction": median_sat}))[:, 1]
df["expected_gain"] = df["churn_prob"] * 0.40 * df["monthly_charge"] * 6 - 300
offer = df[df["expected_gain"] > 0].sort_values("expected_gain", ascending=False)
print(f"\nOffer the voucher to {len(offer)} customers; expected net gain Rs {offer['expected_gain'].sum():,.0f}")
print(f"Offer to all {len(df)} customers instead  : expected net Rs {df['expected_gain'].sum():,.0f}")
print(offer[["customer_id", "plan_type", "tenure_months", "support_calls", "churn_prob", "expected_gain"]].head(5).round(2).to_string(index=False))
offer[["customer_id", "churn_prob", "expected_gain"]].round(3).to_csv(HERE / "retention_call_list.csv", index=False)

# A second predictive question: revenue for the next three months (straight-line trend)
m = pd.read_csv(HERE / "swifttel_monthly.csv")
slope, intercept = np.polyfit(m["t"], m["revenue_lakh"], 1)
print(f"\nRevenue trend: revenue = {intercept:.2f} + {slope:.3f} x t   (t = 1 for Jan 2023)")
for t, name in [(25, "Jan 2025"), (26, "Feb 2025"), (27, "Mar 2025")]:
    print(f"  Forecast {name}: Rs {intercept + slope * t:.1f} lakh")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
(df.groupby(calls_band, observed=True)["churn_flag"].mean() * 100).plot.bar(ax=axes[0], color="#e8833a", rot=0)
axes[0].set_title("Churn % by support calls"); axes[0].set_xlabel("support calls")
axes[1].plot(m["t"], m["revenue_lakh"], marker="o")
axes[1].plot([1, 27], [intercept + slope, intercept + slope * 27], "--", color="grey")
axes[1].set_title("Revenue (Rs lakh) and trend"); axes[1].set_xlabel("month number t")
fig.tight_layout()
fig.savefig(HERE / "swifttel_dashboard.png", dpi=120)
print("\nSaved retention_call_list.csv and swifttel_dashboard.png")
