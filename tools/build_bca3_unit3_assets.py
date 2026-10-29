"""Builds BCA 3 Predictive Analytics Unit 3 data, charts and the numbers / model exports quoted on the pages.

Regression  : a used-car dataset (generated here, fixed seed)
Classification: SwiftTel churn from Unit 1 (lab/swifttel_customers.csv)
Run from the project root:  python tools/build_bca3_unit3_assets.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
UNIT = ROOT / "courses" / "bca3-ds-ai" / "unit-3"
LAB, IMG = UNIT / "lab", UNIT / "img"
LAB.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(303)

# ============================ used cars (regression) ============================
N = 1200
tier = rng.choice(["Budget", "Mid", "Premium"], N, p=[0.45, 0.38, 0.17])
fuel = rng.choice(["Petrol", "Diesel", "CNG", "Electric"], N, p=[0.55, 0.28, 0.12, 0.05])
trans = np.where(rng.random(N) < np.where(tier == "Premium", 0.7, np.where(tier == "Mid", 0.35, 0.12)), "Automatic", "Manual")
age = np.clip(np.round(rng.gamma(2.6, 2.1, N)), 1, 15).astype(int)
km = np.round(np.clip(age * rng.normal(11500, 3500, N) + rng.normal(0, 4000, N), 3000, 240000), -2).astype(int)
owners = np.clip(1 + rng.poisson(age / 9), 1, 4)
city = rng.choice(["Delhi", "Mumbai", "Bengaluru", "Pune", "Jaipur", "Lucknow"], N)
base = np.select([tier == "Budget", tier == "Mid"], [6.0, 9.5], 18.0)
price = (base - 0.55 * age - 0.012 * km / 1000 + 0.8 * (fuel == "Diesel") - 0.3 * (fuel == "CNG") + 2.5 * (fuel == "Electric")
         + 1.0 * (trans == "Automatic") - 0.35 * (owners - 1) + rng.normal(0, 0.8, N))
price = np.round(np.clip(price, 0.8, None), 2)
cars = pd.DataFrame({"listing_id": [f"GB{40001 + i}" for i in range(N)], "brand_tier": tier, "model_year": 2025 - age,
                     "age_years": age, "km_driven": km, "fuel": fuel, "transmission": trans, "owners": owners,
                     "city": city, "price_lakh": price})
cars.to_csv(LAB / "used_cars.csv", index=False)
print("cars", cars.shape, cars.price_lakh.describe().round(2).to_dict())

# simple linear regression: price ~ age
tr, te = train_test_split(cars, test_size=0.3, random_state=42)
slr = LinearRegression().fit(tr[["age_years"]], tr.price_lakh)
p = slr.predict(te[["age_years"]])
print(f"\nSIMPLE  price = {slr.intercept_:.3f} + ({slr.coef_[0]:.3f}) x age   test R2 {r2_score(te.price_lakh, p):.3f}  "
      f"RMSE {mean_squared_error(te.price_lakh, p) ** .5:.3f}  MAE {mean_absolute_error(te.price_lakh, p):.3f}  r {cars.age_years.corr(cars.price_lakh):.3f}")

# multiple regression with dummies
def design(d):
    X = pd.DataFrame({"age_years": d.age_years, "km_thousand": d.km_driven / 1000, "owners": d.owners})
    X["tier_Mid"] = (d.brand_tier == "Mid").astype(int); X["tier_Premium"] = (d.brand_tier == "Premium").astype(int)
    X["fuel_Diesel"] = (d.fuel == "Diesel").astype(int); X["fuel_CNG"] = (d.fuel == "CNG").astype(int); X["fuel_Electric"] = (d.fuel == "Electric").astype(int)
    X["automatic"] = (d.transmission == "Automatic").astype(int)
    return X
mlr = LinearRegression().fit(design(tr), tr.price_lakh)
pm = mlr.predict(design(te))
r2 = r2_score(te.price_lakh, pm); n, k = len(te), design(te).shape[1]
r2_train = r2_score(tr.price_lakh, mlr.predict(design(tr))); n_tr = len(tr)
adj = 1 - (1 - r2_train) * (n_tr - 1) / (n_tr - k - 1)
print(f"MULTIPLE intercept {mlr.intercept_:.3f}  coefs", {c: round(w, 4) for c, w in zip(design(tr).columns, mlr.coef_)})
print(f"  train R2 {r2_train:.3f} adj R2 {adj:.3f} (n={n_tr}, k={k})  test R2 {r2:.3f}  RMSE {mean_squared_error(te.price_lakh, pm) ** .5:.3f}  MAE {mean_absolute_error(te.price_lakh, pm):.3f}")
ex = te.iloc[0]; print("example test car:", ex.to_dict(), " predicted", round(float(mlr.predict(design(te.iloc[[0]]))[0]), 2))
print("CARMODEL", json.dumps({"intercept": float(mlr.intercept_), "coef": {c: float(w) for c, w in zip(design(tr).columns, mlr.coef_)}}))

# ============================ SwiftTel churn (classification) ============================
df = pd.read_csv(ROOT / "courses/bca3-ds-ai/unit-1/lab/swifttel_customers.csv")
X = pd.DataFrame({"tenure_months": df.tenure_months, "monthly_charge": df.monthly_charge, "support_calls": df.support_calls,
                  "prepaid": (df.plan_type == "Prepaid").astype(int), "autopay": (df.autopay == "Yes").astype(int),
                  "satisfaction": df.satisfaction.astype(float)})
y = (df.churned == "Yes").astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
med = Xtr.satisfaction.median(); Xtr = Xtr.fillna({"satisfaction": med}); Xte = Xte.fillna({"satisfaction": med})
print(f"\nCHURN split: train {len(Xtr)} ({ytr.mean():.3f} churn)  test {len(Xte)} ({yte.mean():.3f} churn)  sat median {med}")
sc = StandardScaler().fit(Xtr); Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

models = {
    "Logistic regression": (LogisticRegression(max_iter=5000), False),
    "Decision tree (depth 3)": (DecisionTreeClassifier(max_depth=3, random_state=0), False),
    "KNN (k = 15)": (KNeighborsClassifier(n_neighbors=15), True),
    "Naive Bayes": (GaussianNB(), False),
}
res = {}
for name, (m, scaled) in models.items():
    m.fit(Ztr if scaled else Xtr, ytr)
    pr = m.predict(Zte if scaled else Xte)
    res[name] = dict(acc=accuracy_score(yte, pr), prec=precision_score(yte, pr, zero_division=0), rec=recall_score(yte, pr), flagged=int(pr.sum()))
    print(f"{name:24} acc {res[name]['acc']:.3f}  precision {res[name]['prec']:.3f}  recall {res[name]['rec']:.3f}  predicted churners {pr.sum()}")
print(f"baseline always-No acc {1 - yte.mean():.3f}")
knn_raw = KNeighborsClassifier(n_neighbors=15).fit(Xtr, ytr)
print(f"KNN without scaling acc {accuracy_score(yte, knn_raw.predict(Xte)):.3f}  recall {recall_score(yte, knn_raw.predict(Xte)):.3f}")
for kk in (1, 5, 15, 51):
    kn = KNeighborsClassifier(n_neighbors=kk).fit(Ztr, ytr)
    print(f"  KNN k={kk:2}: train acc {kn.score(Ztr, ytr):.3f} test acc {kn.score(Zte, yte):.3f}")

lg = models["Logistic regression"][0]
print("logistic coefs", {c: round(w, 4) for c, w in zip(X.columns, lg.coef_[0])}, "intercept", round(lg.intercept_[0], 4))
print("odds ratios", {c: round(float(np.exp(w)), 3) for c, w in zip(X.columns, lg.coef_[0])})

tree = models["Decision tree (depth 3)"][0]
print(export_text(tree, feature_names=list(X.columns), show_weights=True))
t = tree.tree_
def node(i):
    if t.children_left[i] == -1:
        v = t.value[i][0]; tot = t.n_node_samples[i]
        return {"leaf": True, "n": int(tot), "rate": round(float(v[1] / v.sum()), 4)}
    return {"leaf": False, "feature": X.columns[t.feature[i]], "thr": round(float(t.threshold[i]), 2), "n": int(t.n_node_samples[i]),
            "gini": round(float(t.impurity[i]), 4), "left": node(t.children_left[i]), "right": node(t.children_right[i])}
print("TREE", json.dumps(node(0)))
root_g = t.impurity[0]; print("root gini", round(root_g, 4), "train churn", round(ytr.mean(), 4))
fi = dict(zip(X.columns, np.round(tree.feature_importances_, 3))); print("tree importances", fi)

# Naive Bayes by counting (categorical version, used in the worked example and the calculator)
d = df.assign(calls3=df.support_calls >= 3, prepaid=df.plan_type == "Prepaid", autopay=df.autopay == "Yes", new=df.tenure_months <= 12)
cnt = {"churn": int((d.churned == "Yes").sum()), "stay": int((d.churned == "No").sum())}
for f in ["calls3", "prepaid", "autopay", "new"]:
    cnt[f] = {"churn": int(d.loc[d.churned == "Yes", f].sum()), "stay": int(d.loc[d.churned == "No", f].sum())}
print("NBCOUNTS", json.dumps(cnt))
sub = d[d.calls3 & d.prepaid & ~d.autopay & d.new]
print("actual churn among calls>=3 & prepaid & no autopay & tenure<=12:", len(sub), round((sub.churned == "Yes").mean(), 3))

# ============================ charts ============================
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 130})
BLUE, ORANGE, GREY, GREEN, RED, VIOLET = "#2a6fdb", "#e8833a", "#9aa4b2", "#1f9d6b", "#d64545", "#7a5af5"
def save(fig, name):
    fig.tight_layout(); fig.savefig(IMG / name, facecolor="white"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.6))
for tname, col in [("Budget", GREY), ("Mid", BLUE), ("Premium", VIOLET)]:
    s = cars[cars.brand_tier == tname]
    ax.scatter(s.age_years + rng.normal(0, .12, len(s)), s.price_lakh, s=9, alpha=.5, color=col, label=tname)
xs = np.array([1, 15]); ax.plot(xs, slr.intercept_ + slr.coef_[0] * xs, color=ORANGE, lw=2.5, label="simple regression line")
ax.set_xlabel("age (years)"); ax.set_ylabel("price (₹ lakh)"); ax.legend(frameon=False, fontsize=9, ncol=2)
ax.set_title("Price falls with age, but brand tier matters too")
save(fig, "car_scatter_line.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
axes[0].scatter(te.price_lakh, pm, s=10, alpha=.5, color=BLUE)
lim = [0, max(te.price_lakh.max(), pm.max()) + 1]; axes[0].plot(lim, lim, color=ORANGE, lw=1.5, ls="--")
axes[0].set_xlabel("actual price (₹ lakh)"); axes[0].set_ylabel("predicted"); axes[0].set_title(f"Multiple regression: test R² = {r2:.2f}")
axes[1].scatter(pm, te.price_lakh - pm, s=10, alpha=.5, color=GREEN); axes[1].axhline(0, color=ORANGE, lw=1.5)
axes[1].set_xlabel("predicted price"); axes[1].set_ylabel("residual (actual − predicted)"); axes[1].set_title("Residuals: no pattern = good")
save(fig, "car_fit_residuals.png")

fig, ax = plt.subplots(figsize=(11.5, 5.2))
plot_tree(tree, feature_names=list(X.columns), class_names=["stay", "churn"], filled=True, impurity=True, proportion=False, rounded=True, fontsize=8, ax=ax)
ax.set_title("Decision tree (depth 3) trained on 1,050 customers")
save(fig, "tree_plot.png")

# KNN decision regions on two scaled features
two = ["tenure_months", "support_calls"]
s2 = StandardScaler().fit(Xtr[two]); kn2 = KNeighborsClassifier(n_neighbors=15).fit(s2.transform(Xtr[two]), ytr)
gx, gy = np.meshgrid(np.linspace(1, 72, 220), np.linspace(-0.5, 7.5, 160))
zz = kn2.predict_proba(s2.transform(pd.DataFrame({"tenure_months": gx.ravel(), "support_calls": gy.ravel()})))[:, 1].reshape(gx.shape)
fig, ax = plt.subplots(figsize=(7, 3.8))
cs = ax.contourf(gx, gy, zz, levels=[0, .2, .35, .5, 1], colors=["#e8f1fd", "#c9dcf8", "#f7d9c2", "#f0b48a"], alpha=.9)
for k, col in [(0, BLUE), (1, RED)]:
    sel = Xtr[ytr == k]
    ax.scatter(sel.tenure_months, sel.support_calls + rng.normal(0, .12, len(sel)), s=7, alpha=.5, color=col, label="churned" if k else "stayed")
ax.set_xlabel("tenure (months)"); ax.set_ylabel("support calls"); ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_title("KNN (k = 15): shaded by predicted churn probability")
save(fig, "knn_regions.png")

calls = np.arange(0, 10)
base_row = pd.DataFrame({"tenure_months": [12] * 10, "monthly_charge": [399] * 10, "support_calls": calls, "prepaid": [1] * 10, "autopay": [0] * 10, "satisfaction": [3.0] * 10})
pc = lg.predict_proba(base_row)[:, 1]
fig, ax = plt.subplots(figsize=(6.6, 3.2))
ax.plot(calls, pc, marker="o", color=RED, lw=2); ax.axhline(.5, color=GREY, ls="--", lw=1)
ax.set_xlabel("support calls"); ax.set_ylabel("P(churn)"); ax.set_ylim(0, 1)
ax.set_title("Logistic regression: the S-curve"); save(fig, "logistic_curve.png")
print("logistic curve p for calls 0..9:", np.round(pc, 3).tolist())

names = list(res)
fig, ax = plt.subplots(figsize=(7.4, 3.3))
xx = np.arange(len(names))
ax.bar(xx - .2, [res[n]["acc"] * 100 for n in names], .4, color=BLUE, label="accuracy")
ax.bar(xx + .2, [res[n]["rec"] * 100 for n in names], .4, color=ORANGE, label="recall (churners caught)")
for i, n_ in enumerate(names):
    ax.text(i - .2, res[n_]["acc"] * 100 + 1.5, f"{res[n_]['acc'] * 100:.0f}", ha="center", fontsize=9)
    ax.text(i + .2, res[n_]["rec"] * 100 + 1.5, f"{res[n_]['rec'] * 100:.0f}", ha="center", fontsize=9)
ax.set_xticks(xx, [n.replace(" (", "\n(") for n in names], fontsize=9); ax.set_ylabel("% on 450 test customers"); ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=9); ax.set_title("Four classifiers, same test set")
save(fig, "model_compare.png")

# the 40 Mid-tier cars used in the in-browser line-fitting lab
mid = cars[cars.brand_tier == "Mid"].head(40)
b, a = np.polyfit(mid.age_years, mid.price_lakh, 1)
pred = a + b * mid.age_years
print(f"\nMID40 least squares: price = {a:.3f} + ({b:.3f}) age  SSE {((mid.price_lakh - pred) ** 2).sum():.2f}  R2 {1 - ((mid.price_lakh - pred) ** 2).sum() / ((mid.price_lakh - mid.price_lakh.mean()) ** 2).sum():.3f}")
print("charts written")
