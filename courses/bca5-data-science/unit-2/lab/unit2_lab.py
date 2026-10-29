"""
Unit 2 Lab · Statistics for Data Science
Descriptive statistics, the normal distribution, sampling and four hypothesis tests (Z, T, Chi-square, ANOVA)
on KiranaMart's 4-week delivery trial.

Datasets: kiranamart_delivery_trial.csv  (600 orders, March 2026)
          atta_pack_weights.csv          (40 packets labelled 1 kg)
Needs   : numpy, pandas, scipy, matplotlib   ->  pip install numpy pandas scipy matplotlib
Run     : python unit2_lab.py                (keep both CSVs in the same folder)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()  # works in Jupyter too
pd.set_option("display.width", 120)
ALPHA = 0.05                       # significance level used for every test below


def decide(p, alpha=ALPHA):
    """Turn a p-value into a plain-English decision."""
    if p < alpha:
        return f"p = {p:.4g} < {alpha}  ->  reject H0 (the difference is statistically significant)"
    return f"p = {p:.4g} >= {alpha}  ->  fail to reject H0 (not enough evidence)"


# ============================================================
# STEP 1 · LOAD AND LOOK
# ============================================================
# The operations team ran a 4-week trial: 3 delivery partners, an old and a new route app.
# Questions from the manager:
#   Q1. What does a "typical" order look like?            (descriptive statistics)
#   Q2. How likely is a delivery to take over 45 minutes?  (normal distribution)
#   Q3. Is the atta packing machine under-filling?         (Z-test)
#   Q4. Does the new route app really save time?           (T-test)
#   Q5. Do new and returning customers pay differently?    (Chi-square test)
#   Q6. Are the three delivery partners equally fast?      (one-way ANOVA)
df = pd.read_csv(HERE / "kiranamart_delivery_trial.csv")
packs = pd.read_csv(HERE / "atta_pack_weights.csv")
print("Orders:", df.shape, "  Packets:", packs.shape)
print(df.head(3))

# ============================================================
# STEP 2 · DESCRIPTIVE STATISTICS (Q1)
# ============================================================
# Centre: mean, median, mode.  Spread: range, variance, standard deviation.
# pandas .var() and .std() divide by (n - 1) by default: the SAMPLE formulas.
v = df["order_value"]
print("Order value  mean:", round(v.mean(), 2), " median:", v.median(), " mode of items:", df["items"].mode()[0])
print("             range:", v.max() - v.min(), " variance:", round(v.var(), 1), " std:", round(v.std(), 2))
print("             skewness:", round(v.skew(), 2), "(> 0 means a long right tail, so mean > median)")
print("Most used payment mode (mode of a category):", df["payment_mode"].mode()[0])
print(df[["order_value", "delivery_min", "items"]].describe().round(2))

# NumPy gives the population formulas unless you ask for ddof=1
d = df["delivery_min"].to_numpy()
print("\nDelivery std  numpy default (ddof=0):", round(np.std(d), 3), "  sample (ddof=1):", round(np.std(d, ddof=1), 3))

# ============================================================
# STEP 3 · NORMAL DISTRIBUTION (Q2)
# ============================================================
# If delivery time is roughly normal, the 68-95-99.7 rule and z-scores apply.
mu, sd = d.mean(), d.std(ddof=1)
for k in (1, 2, 3):
    print(f"Within ±{k} sd: {np.mean(np.abs(d - mu) <= k * sd):.1%}")
z45 = (45 - mu) / sd
print(f"z for 45 min = (45 - {mu:.2f}) / {sd:.2f} = {z45:.2f}")
print(f"P(delivery > 45 min)  normal model: {stats.norm.sf(z45):.2%}   actual data: {np.mean(d > 45):.2%}")

# ============================================================
# STEP 4 · PROBABILITY AND SAMPLING
# ============================================================
# Probability = favourable / total.  Conditional probability: P(late | partner).
late = df["delivery_min"] > 40
print("P(late) =", late.sum(), "/", len(df), "=", round(late.mean(), 3))
print((df.assign(late=late).groupby("partner")["late"].mean() * 100).round(1).rename("% late"))

# The 600 orders are our population. Draw samples of 30 and watch the sample mean.
rng = np.random.default_rng(1)
means = [rng.choice(v, 30, replace=False).mean() for _ in range(2000)]
print("\nPopulation mean:", round(v.mean(), 2), "  mean of 2,000 sample means:", round(np.mean(means), 2))
print("Standard error  sigma / sqrt(n):", round(v.std(ddof=0) / np.sqrt(30), 2), "  observed sd of sample means:", round(np.std(means), 2))

# ============================================================
# STEP 5 · Z-TEST (Q3)
# ============================================================
# H0: mu = 1000 g   H1: mu != 1000 g.  The machine maker states sigma = 12 g (known), n = 40 -> Z-test.
w = packs["weight_g"]
z = (w.mean() - 1000) / (12 / np.sqrt(len(w)))
p = 2 * stats.norm.sf(abs(z))
print(f"Sample mean = {w.mean():.2f} g,  z = {z:.3f}")
print(decide(p))

# ============================================================
# STEP 6 · T-TEST (Q4)
# ============================================================
# H0: mean delivery time is the same with both apps.  Population sigma unknown -> T-test.
# Welch's version (equal_var=False) does not assume equal spread: a safe default.
old = df.loc[df["route_app"] == "Old", "delivery_min"]
new = df.loc[df["route_app"] == "New", "delivery_min"]
t, p = stats.ttest_ind(old, new, equal_var=False)
print(f"Old app {old.mean():.2f} min (n={len(old)}),  New app {new.mean():.2f} min (n={len(new)}),  saving {old.mean() - new.mean():.2f} min")
print(f"t = {t:.3f}");  print(decide(p))

# Same test on order value: the app should NOT change what people buy
t2, p2 = stats.ttest_ind(df.loc[df.route_app == "Old", "order_value"], df.loc[df.route_app == "New", "order_value"], equal_var=False)
print(f"\nOrder value by app: t = {t2:.3f}");  print(decide(p2))

# ============================================================
# STEP 7 · CHI-SQUARE TEST OF INDEPENDENCE (Q5)
# ============================================================
# Two CATEGORICAL columns. H0: payment mode is independent of customer type.
table = pd.crosstab(df["customer_type"], df["payment_mode"])[["UPI", "Card", "Cash on Delivery"]]
print(table)
chi2, p, dof, expected = stats.chi2_contingency(table)
print("Expected counts if H0 were true:\n", np.round(expected, 1))
print(f"chi2 = {chi2:.2f},  dof = {dof}");  print(decide(p))

# ============================================================
# STEP 8 · ONE-WAY ANOVA (Q6)
# ============================================================
# One numeric column, three or more groups. H0: all partners have the same mean delivery time.
print(df.groupby("partner")["delivery_min"].agg(["count", "mean", "std"]).round(2))
groups = [g["delivery_min"] for _, g in df.groupby("partner")]
F, p = stats.f_oneway(*groups)
print(f"F = {F:.2f}");  print(decide(p))

# A control check: does the CITY change delivery time?
F2, p2 = stats.f_oneway(*[g["delivery_min"] for _, g in df.groupby("city")])
print(f"\nBy city: F = {F2:.2f}");  print(decide(p2))

# ============================================================
# STEP 9 · REPORT FOR THE MANAGER
# ============================================================
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
ax[0, 0].hist(v, bins=30, color="#9aa4b2"); ax[0, 0].axvline(v.mean(), color="red", label="mean"); ax[0, 0].axvline(v.median(), color="blue", ls="--", label="median")
ax[0, 0].set_title("Order value (₹): right-skewed"); ax[0, 0].legend()
ax[0, 1].boxplot(groups, tick_labels=[n for n, _ in df.groupby("partner")]); ax[0, 1].set_title(f"Delivery by partner (ANOVA p = {p:.1e})")
ax[1, 0].hist([old, new], bins=20, label=["Old app", "New app"]); ax[1, 0].legend(); ax[1, 0].set_title("Old vs new route app")
(table.div(table.sum(axis=1), axis=0) * 100).plot.bar(stacked=True, ax=ax[1, 1], rot=0); ax[1, 1].set_title("Payment mode by customer type (%)")
fig.tight_layout()
fig.savefig(HERE / "unit2_stats_report.png", dpi=120)
print("Saved unit2_stats_report.png")

print("\nRECOMMENDATIONS")
print("1. Roll out the new route app: it saves about 2.5 minutes per order and the saving is not luck (p < 0.001).")
print("2. Review ExpressGo's contract: it is about 4 minutes slower than SwiftShip, and ANOVA shows partners really differ.")
print("3. Keep Cash on Delivery for first-time customers: 37% of them use it, against 14% of returning customers.")
print("4. Recalibrate the atta packing machine: packets average 995 g, significantly under the 1 kg label.")
