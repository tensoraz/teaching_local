"""Builds the Unit 2 (Statistics for Data Science) datasets and the chart images used on the study page.

Run from the project root:  python tools/build_unit2_assets.py
Output is deterministic (fixed seed), so re-running gives identical files.
The numbers it prints at the end are the ones quoted on the study page, slides and quiz.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
UNIT = ROOT / "courses" / "bca5-data-science" / "unit-2"
LAB, IMG = UNIT / "lab", UNIT / "img"
LAB.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(2602)

# ---------------- 1. delivery trial: 600 orders, March 2026 ----------------
N = 600
cities = ["Delhi", "Mumbai", "Bengaluru", "Pune"]
partners = {"SwiftShip": 0.0, "QuickDrop": 1.5, "ExpressGo": 4.5}   # extra minutes per partner
dates = pd.to_datetime("2026-03-02") + pd.to_timedelta(rng.integers(0, 28, N), unit="D")

rows = []
for i in range(N):
    partner = rng.choice(list(partners), p=[0.36, 0.34, 0.30])
    app = rng.choice(["Old", "New"])
    ctype = rng.choice(["New", "Returning"], p=[0.4, 0.6])
    # new customers trust Cash on Delivery more: a real association for the chi-square test
    pay_p = [0.45, 0.20, 0.35] if ctype == "New" else [0.58, 0.25, 0.17]
    delivery = rng.normal(31.0, 6.0) + partners[partner] - (2.5 if app == "New" else 0.0)
    items = int(min(12, 1 + rng.poisson(2.2)))
    value = float(np.round(np.exp(rng.normal(np.log(520), 0.55)) + 25 * items, -1))   # right-skewed
    rating = int(np.clip(round(rng.normal(6.0 - 0.07 * delivery, 0.75)), 1, 5))
    rows.append({
        "order_id": f"KD{20001 + i}",
        "order_date": dates[i].strftime("%Y-%m-%d"),
        "city": rng.choice(cities),
        "partner": partner,
        "route_app": app,
        "customer_type": ctype,
        "payment_mode": rng.choice(["UPI", "Card", "Cash on Delivery"], p=pay_p),
        "items": items,
        "order_value": value,
        "delivery_min": round(float(np.clip(delivery, 12, 60)), 1),
        "rating": rating,
    })
df = pd.DataFrame(rows).sort_values(["order_date", "order_id"]).reset_index(drop=True)
df.to_csv(LAB / "kiranamart_delivery_trial.csv", index=False)

# ---------------- 2. atta packets: 40 packs labelled 1 kg ----------------
packs = pd.DataFrame({
    "pack_id": [f"AT{101 + i}" for i in range(40)],
    "weight_g": np.round(rng.normal(995.5, 12.0, 40), 1),
})
packs.to_csv(LAB / "atta_pack_weights.csv", index=False)
print("datasets:", df.shape, packs.shape)

# ---------------- charts ----------------
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 130,
})
BLUE, ORANGE, GREEN, VIOLET, GREY, RED = "#2a6fdb", "#e8833a", "#1f9d6b", "#7a5af5", "#9aa4b2", "#d64545"

def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / name, facecolor="white")
    plt.close(fig)

# 1. order value: right-skewed, mean > median
v = df["order_value"]
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(v.clip(upper=2500), bins=30, color=GREY, edgecolor="white")   # one ₹3,890 basket is drawn in the last bar
ax.axvline(v.mean(), color=RED, lw=2.2, label=f"mean ₹{v.mean():.0f}")
ax.axvline(v.median(), color=BLUE, lw=2.2, ls="--", label=f"median ₹{v.median():.0f}")
ax.legend(frameon=False)
ax.set_title("Order value is right-skewed: a few big baskets pull the mean up")
ax.set_xlabel("order value (₹)")
ax.set_ylabel("orders")
save(fig, "order_value_skew.png")

# 2. delivery time vs the normal curve, with ±1σ and ±2σ bands
d = df["delivery_min"]
mu, sd = d.mean(), d.std()
fig, ax = plt.subplots(figsize=(7, 3.3))
ax.hist(d, bins=28, density=True, color=BLUE, alpha=.35, edgecolor="white")
x = np.linspace(d.min() - 3, d.max() + 3, 300)
ax.plot(x, stats.norm.pdf(x, mu, sd), color=BLUE, lw=2.4)
ax.axvspan(mu - 2 * sd, mu + 2 * sd, color=GREEN, alpha=.07)
ax.axvspan(mu - sd, mu + sd, color=GREEN, alpha=.12)
for k in (-2, -1, 0, 1, 2):
    ax.axvline(mu + k * sd, color=GREY, lw=1, ls=":")
ax.text(mu, ax.get_ylim()[1] * .92, "±1σ: 68%", ha="center", color=GREEN, fontweight="bold")
ax.text(mu + 1.5 * sd, ax.get_ylim()[1] * .55, "±2σ: 95%", ha="center", color=GREEN)
ax.set_title(f"Delivery time is close to normal (μ ≈ {mu:.1f}, σ ≈ {sd:.1f} min)")
ax.set_xlabel("delivery time (minutes)")
ax.set_yticks([])
save(fig, "delivery_normal.png")

# 3. central limit theorem: means of samples of 30 order values
means = [v.sample(30, random_state=s).mean() for s in range(2000)]
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.0))
axs[0].hist(v.clip(upper=2500), bins=30, color=GREY, edgecolor="white")
axs[0].set_title("Population: 600 order values")
axs[0].set_yticks([])
axs[1].hist(means, bins=30, color=VIOLET, alpha=.8, edgecolor="white")
axs[1].set_title("2,000 sample means (n = 30)")
axs[1].set_yticks([])
for a in axs:
    a.set_xlabel("₹")
    a.set_xlim(0, 2600)   # same scale, so the narrowing is visible
save(fig, "clt_sample_means.png")

# 4. delivery time by partner (ANOVA)
order = list(partners)
fig, ax = plt.subplots(figsize=(6.6, 3.2))
bp = ax.boxplot([df.loc[df.partner == p, "delivery_min"] for p in order], tick_labels=order, patch_artist=True, widths=.5,
                medianprops=dict(color="black"))
for patch, c in zip(bp["boxes"], [GREEN, BLUE, ORANGE]):
    patch.set_facecolor(c)
    patch.set_alpha(.55)
for i, p in enumerate(order, 1):
    m = df.loc[df.partner == p, "delivery_min"].mean()
    ax.plot(i, m, "D", color="black", ms=5)
    ax.text(i + .28, m, f"{m:.1f}", va="center")
ax.set_ylabel("delivery time (min)")
ax.set_title("Delivery time by partner (◆ = mean)")
save(fig, "partner_boxplot.png")

# 5. old vs new route app (t-test)
fig, ax = plt.subplots(figsize=(6.6, 3.0))
for app, c in [("Old", GREY), ("New", BLUE)]:
    s = df.loc[df.route_app == app, "delivery_min"]
    ax.hist(s, bins=24, alpha=.55, color=c, label=f"{app} app: mean {s.mean():.1f} min")
    ax.axvline(s.mean(), color=c, lw=2.2)
ax.legend(frameon=False)
ax.set_xlabel("delivery time (min)")
ax.set_yticks([])
ax.set_title("Old vs new route app")
save(fig, "route_app_compare.png")

# 6. payment mode by customer type (chi-square)
ct = pd.crosstab(df.customer_type, df.payment_mode, normalize="index")[["UPI", "Card", "Cash on Delivery"]] * 100
fig, ax = plt.subplots(figsize=(6.6, 2.6))
left = np.zeros(len(ct))
for col, c in zip(ct.columns, [BLUE, VIOLET, ORANGE]):
    ax.barh(ct.index, ct[col], left=left, color=c, label=col)
    for y, (l, w) in enumerate(zip(left, ct[col])):
        ax.text(l + w / 2, y, f"{w:.0f}%", ha="center", va="center", color="white", fontweight="bold")
    left += ct[col].values
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.18))
ax.set_xlim(0, 100)
ax.set_xticks([])
ax.set_title("How each customer type pays (% of their orders)")
save(fig, "payment_by_customer.png")

# 7. atta packets (z-test)
w = packs["weight_g"]
fig, ax = plt.subplots(figsize=(6.6, 2.6))
ax.scatter(w, rng.uniform(-.3, .3, len(w)), color=ORANGE, alpha=.8, s=30)
ax.axvline(1000, color=RED, lw=2, label="label: 1000 g")
ax.axvline(w.mean(), color=BLUE, lw=2, ls="--", label=f"sample mean {w.mean():.1f} g")
ax.set_yticks([])
ax.set_ylim(-1, 1)
ax.legend(frameon=False, loc="upper left")
ax.set_xlabel("weight (g)")
ax.set_title("40 atta packets labelled 1 kg")
save(fig, "atta_weights.png")

# ---------------- numbers quoted on the pages ----------------
pd.set_option("display.width", 120)
print("\n--- descriptive ---")
print(df[["order_value", "delivery_min", "items"]].describe().round(2))
print("order_value mean/median/skew:", round(v.mean(), 2), v.median(), round(v.skew(), 2))
print("items mode:", df["items"].mode()[0], " payment mode:", df["payment_mode"].mode()[0])
print("delivery mean/sd/var:", round(mu, 2), round(sd, 2), round(d.var(), 2), " skew:", round(d.skew(), 2))
for k in (1, 2, 3):
    print(f"within ±{k}σ: {((d - mu).abs() <= k * sd).mean():.3f}")
print("P(delivery > 45) data:", round((d > 45).mean(), 4), " normal model:", round(1 - stats.norm.cdf(45, mu, sd), 4))
print("late (> 40 min):", (d > 40).sum(), round((d > 40).mean(), 4))
print(pd.crosstab(df.partner, d > 40, margins=True))
print("SE of mean order value n=30:", round(v.std(ddof=0) / np.sqrt(30), 2), " sd of sample means:", round(np.std(means), 2),
      " mean of means:", round(np.mean(means), 2), " pop mean:", round(v.mean(), 2))

print("\n--- z-test: atta ---")
z = (w.mean() - 1000) / (12 / np.sqrt(len(w)))
print("n, mean, sd:", len(w), round(w.mean(), 2), round(w.std(), 2), " z:", round(z, 3),
      " p two-sided:", round(2 * stats.norm.sf(abs(z)), 4), " p one-sided:", round(stats.norm.cdf(z), 4))

print("\n--- t-test: route app ---")
old, new = df.loc[df.route_app == "Old", "delivery_min"], df.loc[df.route_app == "New", "delivery_min"]
print("n old/new:", len(old), len(new), " means:", round(old.mean(), 2), round(new.mean(), 2),
      " sds:", round(old.std(), 2), round(new.std(), 2))
print(stats.ttest_ind(old, new, equal_var=False))
ov_old, ov_new = df.loc[df.route_app == "Old", "order_value"], df.loc[df.route_app == "New", "order_value"]
print("order_value by app:", round(ov_old.mean(), 1), round(ov_new.mean(), 1), stats.ttest_ind(ov_old, ov_new, equal_var=False))
print("one-sample t, delivery vs 30:", stats.ttest_1samp(d, 30))

print("\n--- chi-square: customer type x payment ---")
tab = pd.crosstab(df.customer_type, df.payment_mode)[["UPI", "Card", "Cash on Delivery"]]
print(tab)
chi2, p, dof, exp = stats.chi2_contingency(tab)
print("chi2:", round(chi2, 3), " dof:", dof, " p:", p)
print(np.round(exp, 1))
print("city x payment:", stats.chi2_contingency(pd.crosstab(df.city, df.payment_mode))[:3])

print("\n--- one-way ANOVA: partner ---")
groups = [df.loc[df.partner == p, "delivery_min"] for p in order]
print(df.groupby("partner")["delivery_min"].agg(["count", "mean", "std"]).round(2))
print(stats.f_oneway(*groups))
print("city ANOVA:", stats.f_oneway(*[df.loc[df.city == c, "delivery_min"] for c in cities]))
print("rating by app:", df.groupby("route_app")["rating"].mean().round(2).to_dict())
print("corr delivery-rating:", round(df["delivery_min"].corr(df["rating"]), 3))
print("counts:", df.partner.value_counts().to_dict(), df.route_app.value_counts().to_dict(), df.customer_type.value_counts().to_dict())
