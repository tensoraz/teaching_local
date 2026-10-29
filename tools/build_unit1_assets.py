"""Builds the Unit 1 practice dataset and the chart images used on the study page.

Run from the project root:  python tools/build_unit1_assets.py
Output is deterministic (fixed seed), so re-running gives identical files.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
UNIT = ROOT / "courses" / "bca5-data-science" / "unit-1"
LAB, IMG = UNIT / "lab", UNIT / "img"
LAB.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(2026)
N = 1200

cities = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Pune", "Chandigarh"]
city_w = [0.22, 0.21, 0.19, 0.15, 0.12, 0.11]
cats = {  # category: (price low, price high, typical qty)
    "Grocery": (40, 600, 4),
    "Electronics": (800, 25000, 1),
    "Fashion": (300, 3500, 2),
    "Home": (200, 6000, 1),
    "Beauty": (150, 1800, 2),
}
cat_w = [0.38, 0.12, 0.22, 0.14, 0.14]
pay = ["UPI", "Card", "Cash on Delivery", "Net Banking"]
pay_w = [0.52, 0.24, 0.17, 0.07]

dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, N), unit="D")
# festive-season boost: more orders in Oct-Nov (Diwali sale)
festive = rng.random(N) < 0.18
dates = np.where(festive, pd.to_datetime("2025-10-05") + pd.to_timedelta(rng.integers(0, 50, N), unit="D"), dates)
dates = pd.to_datetime(dates)

category = rng.choice(list(cats), N, p=cat_w)
rows = []
for i in range(N):
    c = category[i]
    lo, hi, q = cats[c]
    price = float(np.round(np.exp(rng.uniform(np.log(lo), np.log(hi))), -1))
    month = dates[i].month
    discount = int(rng.choice([0, 5, 10, 15, 20, 30, 40], p=[0.3, 0.15, 0.2, 0.12, 0.12, 0.07, 0.04]))
    if month in (10, 11):
        discount = min(50, discount + 10)
    qty = max(1, int(rng.poisson(q) + (discount >= 20)))
    delivery = int(np.clip(rng.normal(3.2, 1.4), 1, 9))
    # ratings drop when delivery is slow - a pattern students can discover
    rating = int(np.clip(round(rng.normal(4.6 - 0.28 * delivery, 0.7)), 1, 5))
    rows.append({
        "order_id": f"KM{10001 + i}",
        "order_date": dates[i].strftime("%Y-%m-%d"),
        "city": rng.choice(cities, p=city_w),
        "category": c,
        "payment_mode": rng.choice(pay, p=pay_w),
        "customer_type": rng.choice(["New", "Returning"], p=[0.37, 0.63]),
        "customer_age": int(np.clip(rng.normal(31, 9), 18, 70)),
        "quantity": qty,
        "unit_price": price,
        "discount_pct": discount,
        "delivery_days": delivery,
        "rating": rating,
    })

df = pd.DataFrame(rows)

# --- deliberately add real-world mess for the "Data Preparation" step ---
miss_rating = rng.choice(N, 64, replace=False)
df.loc[miss_rating, "rating"] = np.nan
miss_age = rng.choice(N, 31, replace=False)
df.loc[miss_age, "customer_age"] = np.nan
df.loc[rng.choice(N, 3, replace=False), "quantity"] = -1          # data-entry errors
df.loc[rng.choice(N, 5, replace=False), "city"] = "delhi "         # inconsistent text
dups = df.sample(8, random_state=7)
df = pd.concat([df, dups]).sort_values("order_date").reset_index(drop=True)
df["customer_age"] = df["customer_age"].astype("Int64")
df["rating"] = df["rating"].astype("Int64")

df.to_csv(LAB / "kiranamart_sales_2025.csv", index=False)
print("dataset:", df.shape)

# ---------------- clean copy used for charts ----------------
clean = df.drop_duplicates().copy()
clean = clean[clean["quantity"] > 0]
clean["city"] = clean["city"].str.strip().str.title()
clean["order_date"] = pd.to_datetime(clean["order_date"])
clean["revenue"] = clean["quantity"] * clean["unit_price"] * (1 - clean["discount_pct"] / 100)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 130,
})
BLUE, ORANGE, GREY = "#2a6fdb", "#e8833a", "#9aa4b2"

def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / name, facecolor="white")
    plt.close(fig)

# 1. missing values per column
miss = df.isna().sum()
miss = miss[miss > 0].sort_values()
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.barh(miss.index, miss.values, color=ORANGE)
for y, v in enumerate(miss.values):
    ax.text(v + 1, y, str(v), va="center")
ax.set_title("Missing values per column (raw data)")
ax.set_xlabel("number of empty cells")
save(fig, "missing_values.png")

# 2. monthly revenue trend
monthly = clean.set_index("order_date")["revenue"].resample("MS").sum() / 1e5
fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.plot(monthly.index.strftime("%b"), monthly.values, marker="o", color=BLUE, lw=2.2)
peak = monthly.idxmax()
ax.annotate("Diwali sale peak", xy=(peak.strftime("%b"), monthly.max()),
            xytext=(-110, -8), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=GREY))
ax.set_title("Monthly revenue, 2025 (₹ lakh)")
ax.set_ylabel("₹ lakh")
ax.grid(axis="y", alpha=.3)
save(fig, "monthly_revenue.png")

# 3. revenue by category
bycat = (clean.groupby("category")["revenue"].sum() / 1e5).sort_values()
fig, ax = plt.subplots(figsize=(6.4, 3))
bars = ax.barh(bycat.index, bycat.values, color=[GREY] * (len(bycat) - 1) + [BLUE])
for b, v in zip(bars, bycat.values):
    ax.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center")
ax.set_title("Revenue by category (₹ lakh)")
save(fig, "revenue_by_category.png")

# 4. payment mode share
pm = clean["payment_mode"].value_counts()
fig, ax = plt.subplots(figsize=(6.4, 2.8))
ax.bar(pm.index, pm.values, color=[BLUE, GREY, GREY, GREY])
for x, v in enumerate(pm.values):
    ax.text(x, v + 5, f"{v / pm.sum():.0%}", ha="center")
ax.set_title("How customers pay (number of orders)")
save(fig, "payment_mode.png")

# 5. delivery days vs average rating
dr = clean.groupby("delivery_days")["rating"].mean()
fig, ax = plt.subplots(figsize=(6.4, 3))
ax.plot(dr.index, dr.values, marker="o", color=ORANGE, lw=2.2)
ax.set_xlabel("delivery time (days)")
ax.set_ylabel("average rating")
ax.set_ylim(1, 5)
ax.grid(alpha=.3)
ax.set_title("Slow delivery → lower ratings")
save(fig, "delivery_vs_rating.png")

# summary numbers the study page quotes
print("clean rows:", len(clean))
print("total revenue (lakh):", round(clean["revenue"].sum() / 1e5, 1))
print(bycat.round(1).to_dict())
print(pm.to_dict())
print(dr.round(2).to_dict())
print(monthly.round(1).to_dict())
print(clean.groupby("city")["revenue"].sum().div(1e5).round(1).sort_values().to_dict())
