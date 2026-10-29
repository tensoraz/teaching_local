"""
Unit 1 Lab · Introduction to Data Science
A complete (mini) data-science project, organised by the CRISP-DM phases.

Dataset : kiranamart_sales_2025.csv  (1,208 orders from a 6-city retail chain)
Needs   : pandas, matplotlib, numpy      ->  pip install pandas matplotlib numpy
Run     : python unit1_lab.py            (keep the CSV in the same folder)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()  # works in Jupyter too
pd.set_option("display.width", 120)

# ============================================================
# PHASE 1 · BUSINESS UNDERSTANDING
# ============================================================
# KiranaMart's manager asks:
#   Q1. Where does our money come from (city, category)?
#   Q2. When do we sell the most?
#   Q3. Why are some customers giving low ratings?
# Success = clear, number-backed answers the manager can act on.

# ============================================================
# PHASE 2 · DATA UNDERSTANDING
# ============================================================
df = pd.read_csv(HERE / "kiranamart_sales_2025.csv")
print("Shape (rows, columns):", df.shape)
print(df.head(3))
print("\nColumn types:\n", df.dtypes)
print("\nMissing values per column:\n", df.isna().sum()[df.isna().sum() > 0])
print("\nDuplicate rows:", df.duplicated().sum())
print("Rows with quantity <= 0:", (df["quantity"] <= 0).sum())
print("City spellings:", sorted(df["city"].unique()))

# ============================================================
# PHASE 3 · DATA PREPARATION
# ============================================================
clean = df.drop_duplicates().copy()                       # 1. remove exact duplicates
clean = clean[clean["quantity"] > 0]                      # 2. drop impossible quantities
clean["city"] = clean["city"].str.strip().str.title()     # 3. fix "delhi " -> "Delhi"
clean["order_date"] = pd.to_datetime(clean["order_date"]) # 4. text -> real dates
clean["month"] = clean["order_date"].dt.month_name().str[:3]
clean["revenue"] = clean["quantity"] * clean["unit_price"] * (1 - clean["discount_pct"] / 100)
# rating is missing for some orders: keep the row, analyse rating only where present
print("\nRows before:", len(df), " after cleaning:", len(clean))

# ============================================================
# PHASE 4 · MODELING (here: analysis + a simple fitted model)
# ============================================================
total = clean["revenue"].sum()
print(f"\nTotal revenue: Rs {total:,.0f}")

by_city = clean.groupby("city")["revenue"].sum().sort_values(ascending=False)
print("\nQ1a. Revenue by city (Rs lakh):\n", (by_city / 1e5).round(1))

by_cat = clean.groupby("category").agg(orders=("order_id", "count"), revenue=("revenue", "sum"))
by_cat["share_of_orders_%"] = (100 * by_cat["orders"] / by_cat["orders"].sum()).round(1)
by_cat["share_of_revenue_%"] = (100 * by_cat["revenue"] / total).round(1)
print("\nQ1b. Category mix:\n", by_cat.sort_values("revenue", ascending=False).drop(columns="revenue"))

monthly = clean.groupby(clean["order_date"].dt.month)["revenue"].sum() / 1e5
print("\nQ2. Monthly revenue (Rs lakh):\n", monthly.round(1).to_string())

rated = clean.dropna(subset=["rating"])
print("\nQ3. Average rating by delivery days:\n", rated.groupby("delivery_days")["rating"].mean().round(2).to_string())

# A first "model": fit a straight line  rating = a * delivery_days + b
a, b = np.polyfit(rated["delivery_days"], rated["rating"], deg=1)
r = np.corrcoef(rated["delivery_days"], rated["rating"])[0, 1]
print(f"\nFitted model: rating = {a:.2f} x delivery_days + {b:.2f}   (correlation r = {r:.2f})")

# ============================================================
# PHASE 5 · EVALUATION
# ============================================================
pred = a * rated["delivery_days"] + b
mae = (rated["rating"] - pred).abs().mean()
print(f"Mean absolute error of the model: {mae:.2f} stars")
for d in (2, 5):
    print(f"  predicted rating if delivered in {d} days: {a * d + b:.2f}")
# Does it answer the business question? Yes: every extra day of delivery costs ~|a| stars.

# ============================================================
# PHASE 6 · DEPLOYMENT (share the result)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
(by_cat["revenue"].sort_values() / 1e5).plot.barh(ax=axes[0], color="#2a6fdb", title="Revenue by category (Rs lakh)")
monthly.plot(ax=axes[1], marker="o", color="#2a6fdb", title="Monthly revenue (Rs lakh)")
rated.groupby("delivery_days")["rating"].mean().plot(ax=axes[2], marker="o", color="#e8833a",
                                                     title="Avg rating vs delivery days")
plt.tight_layout()
plt.savefig(HERE / "kiranamart_dashboard.png", dpi=120)
print("\nSaved dashboard -> kiranamart_dashboard.png")

print("""
RECOMMENDATIONS FOR THE MANAGER
1. Electronics brings about half the revenue from about 1 in 8 orders: protect its stock before Oct-Nov.
2. Oct-Nov (festive sale) is the peak: plan inventory and delivery staff from September.
3. Ratings fall steadily as delivery time grows: target delivery within 3 days.
""")
