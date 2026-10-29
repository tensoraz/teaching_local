"""Builds the BCA 3 Predictive Analytics Unit 1 datasets, charts and model numbers.

Run from the project root:  python tools/build_bca3_unit1_assets.py
Output is deterministic (fixed seed), so re-running gives identical files.
The model coefficients printed at the end are pasted into study.html (in-browser lab).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
UNIT = ROOT / "courses" / "bca3-ds-ai" / "unit-1"
LAB, IMG = UNIT / "lab", UNIT / "img"
LAB.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(2025)
N = 1500

# ---------------- customer snapshot on 1 Jan 2025; churn = left during Jan-Mar 2025 ----------------
plan = rng.choice(["Prepaid", "Postpaid"], N, p=[0.62, 0.38])
prepaid = plan == "Prepaid"
tier = rng.choice(["Tier-1", "Tier-2", "Tier-3"], N, p=[0.45, 0.35, 0.20])
tenure = np.clip(rng.gamma(1.6, 14, N).astype(int), 1, 72)
charge = np.where(prepaid, rng.choice([199, 239, 299, 349, 399, 479, 599], N),
                  rng.choice([399, 499, 599, 699, 799, 999], N))
autopay = np.where(rng.random(N) < np.where(prepaid, 0.15, 0.70), "Yes", "No")
sat = np.clip(np.round(rng.normal(3.6, 1.0, N)), 1, 5).astype(int)
calls = np.clip(rng.poisson(0.6 + 0.45 * (5 - sat)), 0, 9)
data_gb = np.round(np.clip(charge / 14 + rng.normal(0, 6, N), 0.5, None), 1)
age = np.clip(rng.normal(34, 11, N), 18, 70).astype(int)
gender = rng.choice(["Male", "Female"], N, p=[0.55, 0.45])
join = pd.Timestamp("2025-01-01") - pd.to_timedelta(tenure * 30.4 + rng.integers(0, 28, N), unit="D")

z = (-1.05 + 0.38 * calls - 0.04 * tenure + 0.65 * prepaid - 0.8 * (autopay == "Yes")
     - 0.35 * (sat - 3) + 0.25 * (tier == "Tier-3") + 0.0015 * (charge - 450))
churn = np.where(rng.random(N) < 1 / (1 + np.exp(-z)), "Yes", "No")

df = pd.DataFrame({
    "customer_id": [f"ST{10001 + i}" for i in range(N)],
    "gender": gender, "age": age, "city_tier": tier, "plan_type": plan,
    "join_date": join.strftime("%Y-%m-%d"), "tenure_months": tenure, "monthly_charge": charge,
    "data_gb": data_gb, "autopay": autopay, "support_calls": calls,
    "satisfaction": sat.astype(float), "churned": churn,
})
df.loc[rng.choice(N, 45, replace=False), "satisfaction"] = np.nan   # 45 customers skipped the survey
df["satisfaction"] = df["satisfaction"].astype("Int64")
df.to_csv(LAB / "swifttel_customers.csv", index=False)

# ---------------- company-level monthly series 2023-2024 (for the Excel forecast) ----------------
t = np.arange(1, 25)
months = pd.date_range("2023-01-01", periods=24, freq="MS")
active = np.round(118000 + 1450 * t + rng.normal(0, 900, 24)).astype(int)
new = np.round(active * rng.uniform(0.032, 0.040, 24)).astype(int)
lost = np.round(active * rng.uniform(0.021, 0.027, 24)).astype(int)
revenue = np.round(active * 405 / 1e5 + rng.normal(0, 6, 24), 1)          # Rs lakh
monthly = pd.DataFrame({"month": months.strftime("%Y-%m"), "t": t, "active_customers": active,
                        "new_customers": new, "churned_customers": lost, "revenue_lakh": revenue})
monthly.to_csv(LAB / "swifttel_monthly.csv", index=False)

# ---------------- numbers quoted on the pages ----------------
print("customers:", df.shape, " churn rate:", round((df.churned == "Yes").mean() * 100, 1), "%")
print("missing:", df.isna().sum()[df.isna().sum() > 0].to_dict())
print("\nby plan:\n", df.groupby("plan_type")["churned"].apply(lambda s: round((s == "Yes").mean() * 100, 1)))
print("\nby autopay:\n", df.groupby("autopay")["churned"].apply(lambda s: round((s == "Yes").mean() * 100, 1)))
print("\nby tier:\n", df.groupby("city_tier")["churned"].apply(lambda s: round((s == "Yes").mean() * 100, 1)))
calls_band = pd.cut(df.support_calls, [-1, 0, 1, 2, 3, 9], labels=["0", "1", "2", "3", "4+"])
print("\nby support calls:\n", df.groupby(calls_band, observed=True)["churned"].agg(
    customers="size", churn_pct=lambda s: round((s == "Yes").mean() * 100, 1)))
ten_band = pd.cut(df.tenure_months, [0, 6, 12, 24, 48, 72], labels=["1-6", "7-12", "13-24", "25-48", "49-72"])
print("\nby tenure:\n", df.groupby(ten_band, observed=True)["churned"].agg(
    customers="size", churn_pct=lambda s: round((s == "Yes").mean() * 100, 1)))
print("\nchurners vs stayers (means):\n", df.groupby("churned")[["tenure_months", "support_calls", "monthly_charge", "satisfaction"]].mean().round(2))
print("\nsummary:\n", df[["age", "tenure_months", "monthly_charge", "data_gb", "support_calls"]].describe().round(1))

slope, intercept = np.polyfit(monthly.t, monthly.revenue_lakh, 1)
fc = {k: round(intercept + slope * k, 1) for k in (25, 26, 27)}
print(f"\nmonthly revenue trend: revenue = {intercept:.2f} + {slope:.3f} * t ; forecast {fc}")
print(monthly.head(3).to_string(index=False)); print(monthly.tail(2).to_string(index=False))

# ---------------- predictive model (preview of Units 3-4) ----------------
X = pd.DataFrame({
    "tenure_months": df.tenure_months, "monthly_charge": df.monthly_charge,
    "support_calls": df.support_calls, "prepaid": (df.plan_type == "Prepaid").astype(int),
    "autopay": (df.autopay == "Yes").astype(int), "tier3": (df.city_tier == "Tier-3").astype(int),
    "satisfaction": df.satisfaction.astype(float),
})
y = (df.churned == "Yes").astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
med = Xtr.satisfaction.median()
Xtr, Xte = Xtr.fillna({"satisfaction": med}), Xte.fillna({"satisfaction": med})
model = LogisticRegression(max_iter=5000).fit(Xtr, ytr)
acc = (model.predict(Xte) == yte).mean()
base = 1 - yte.mean()
p_te = model.predict_proba(Xte)[:, 1]
top = pd.Series(yte.values)[np.argsort(-p_te)[: len(yte) // 10]]
print(f"\ntest accuracy {acc:.3f}  baseline (always No) {base:.3f}  "
      f"top-10% churn rate {top.mean():.3f}  overall {yte.mean():.3f}  lift {top.mean() / yte.mean():.2f}")
coef = dict(zip(X.columns, np.round(model.coef_[0], 4)))
print("MODEL", json.dumps({"intercept": round(float(model.intercept_[0]), 4), "coef": {k: float(v) for k, v in coef.items()},
                           "sat_median": float(med)}))

# prescriptive: offer Rs 300 to customers where expected benefit > cost
Xall = X.fillna({"satisfaction": med})
p_all = model.predict_proba(Xall)[:, 1]
value = df.monthly_charge * 6                     # revenue kept if the customer stays 6 more months
ev = p_all * 0.40 * value - 300                   # 40% of would-be churners accept and stay
print(f"\nprescriptive: target {int((ev > 0).sum())} customers, expected net Rs {ev[ev > 0].sum():,.0f}; "
      f"offer to everyone: Rs {ev.sum():,.0f}")

# risk deciles for the lift chart
dec = pd.qcut(pd.Series(p_all).rank(method="first", ascending=False), 10, labels=range(1, 11))
dec_rate = pd.Series(y.values).groupby(dec.values, observed=True).mean() * 100
print("\nchurn % by risk decile:", dec_rate.round(1).to_dict())

# ---------------- charts ----------------
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 130,
})
BLUE, ORANGE, GREY, GREEN = "#2a6fdb", "#e8833a", "#9aa4b2", "#1f9d6b"

def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / name, facecolor="white")
    plt.close(fig)

rate = df.groupby(calls_band, observed=True)["churned"].apply(lambda s: (s == "Yes").mean() * 100)
fig, ax = plt.subplots(figsize=(6.4, 3))
ax.bar(rate.index.astype(str), rate.values, color=[GREY, GREY, GREY, ORANGE, ORANGE])
for x, v in enumerate(rate.values):
    ax.text(x, v + 1, f"{v:.0f}%", ha="center")
ax.set_xlabel("support calls in the last 3 months"); ax.set_ylabel("% who churned")
ax.set_title("More complaints → more churn")
save(fig, "churn_by_calls.png")

rate = df.groupby(ten_band, observed=True)["churned"].apply(lambda s: (s == "Yes").mean() * 100)
fig, ax = plt.subplots(figsize=(6.4, 3))
ax.bar(rate.index.astype(str), rate.values, color=[ORANGE, ORANGE, GREY, GREY, GREY])
for x, v in enumerate(rate.values):
    ax.text(x, v + 1, f"{v:.0f}%", ha="center")
ax.set_xlabel("tenure (months with SwiftTel)"); ax.set_ylabel("% who churned")
ax.set_title("First-year customers leave most often")
save(fig, "churn_by_tenure.png")

fig, ax = plt.subplots(figsize=(7.2, 3.3))
ax.plot(monthly.t, monthly.revenue_lakh, marker="o", color=BLUE, lw=2, label="actual")
tt = np.arange(1, 28)
ax.plot(tt, intercept + slope * tt, color=GREY, ls="--", lw=1.5, label="trend line")
ax.plot(list(fc), list(fc.values()), marker="s", color=ORANGE, lw=0, ms=8, label="forecast Jan–Mar 2025")
ax.set_xticks([1, 7, 13, 19, 25], ["Jan 23", "Jul 23", "Jan 24", "Jul 24", "Jan 25"])
ax.set_ylabel("₹ lakh"); ax.set_title("Monthly revenue and a straight-line forecast")
ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", alpha=.3)
save(fig, "revenue_forecast.png")

fig, ax = plt.subplots(figsize=(6.6, 3.1))
ax.bar(dec_rate.index.astype(str), dec_rate.values, color=[GREEN] + [GREY] * 9)
ax.axhline(y.mean() * 100, color=ORANGE, ls="--", lw=1.5)
ax.text(9.4, y.mean() * 100 + 1.5, f"average {y.mean() * 100:.0f}%", ha="right", color=ORANGE, fontsize=9)
for x, v in enumerate(dec_rate.values):
    ax.text(x, v + 1, f"{v:.0f}", ha="center", fontsize=9)
ax.set_xlabel("risk group (1 = the 10% the model rates most likely to leave)"); ax.set_ylabel("% who churned")
ax.set_title("The model ranks risky customers first")
save(fig, "risk_deciles.png")
print("\ncharts written to", IMG)
