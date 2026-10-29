"""Builds the BCA 3 Predictive Analytics Unit 2 raw datasets, charts and the numbers quoted on the pages.

The three messy source files are made FROM Unit 1's clean customer file, so students see
where a clean dataset comes from. Run from the project root after Unit 1's builder:
    python tools/build_bca3_unit2_assets.py
Deterministic (fixed seed).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "courses" / "bca3-ds-ai" / "unit-1" / "lab" / "swifttel_customers.csv"
UNIT = ROOT / "courses" / "bca3-ds-ai" / "unit-2"
LAB, IMG = UNIT / "lab", UNIT / "img"
LAB.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

base = pd.read_csv(SRC)
N = len(base)

# ======================= source 1: CRM export (messy) =======================
crm = base[["customer_id", "gender", "age", "city_tier", "plan_type", "join_date", "monthly_charge",
            "autopay", "satisfaction", "churned"]].copy().astype(object)

def variant(value, table):
    opts, p = table[value]
    return rng.choice(opts, p=p)

g_tab = {"Male": (["Male", "M", "male"], [.82, .12, .06]), "Female": (["Female", "F", "female "], [.82, .12, .06])}
p_tab = {"Prepaid": (["Prepaid", "prepaid", "Pre-paid", "PREPAID"], [.85, .07, .05, .03]),
         "Postpaid": (["Postpaid", "postpaid", "Post-paid"], [.88, .07, .05])}
crm["gender"] = [variant(v, g_tab) for v in base.gender]
crm["plan_type"] = [variant(v, p_tab) for v in base.plan_type]

# age: missing at random GIVEN plan (prepaid KYC forms are often incomplete) -> MAR
miss_age = rng.random(N) < np.where(base.plan_type == "Prepaid", 0.065, 0.015)
crm.loc[miss_age, "age"] = np.nan
bad_age_idx = rng.choice(np.where(~miss_age)[0], 5, replace=False)
crm.loc[bad_age_idx, "age"] = [0, 0, 0, 151, 234]                     # impossible ages

# satisfaction: unhappy customers skip the survey more often -> MNAR (on top of Unit 1's 45 random gaps)
s = base.satisfaction
mnar = s.notna() & (rng.random(N) < np.where(s <= 2, 0.45, np.where(s == 3, 0.06, 0.01)))
crm.loc[mnar, "satisfaction"] = np.nan

# monthly charge typed by hand: symbols, decimals, and a few x100 typos
ch = base.monthly_charge.astype(int)
style = rng.choice(["plain", "rupee", "decimal"], N, p=[.85, .10, .05])
txt = np.where(style == "rupee", "₹" + ch.astype(str), np.where(style == "decimal", ch.astype(str) + ".00", ch.astype(str)))
typo_idx = rng.choice(N, 4, replace=False)
txt[typo_idx] = (ch.iloc[typo_idx] * 100).astype(str)
crm["monthly_charge"] = txt

# dates in two formats
jd = pd.to_datetime(base.join_date)
crm["join_date"] = np.where(rng.random(N) < 0.2, jd.dt.strftime("%d/%m/%Y"), jd.dt.strftime("%Y-%m-%d"))
crm["churned"] = np.where(rng.random(N) < 0.02, base.churned.str.lower(), base.churned)   # a few "yes"/"no"
crm["age"] = crm["age"].map(lambda v: "" if pd.isna(v) else str(int(v)))
crm["satisfaction"] = crm["satisfaction"].map(lambda v: "" if pd.isna(v) else str(int(v)))
dups = crm.sample(15, random_state=3)
crm = pd.concat([crm, dups]).sample(frac=1, random_state=4).reset_index(drop=True)
crm.to_csv(LAB / "crm_customers.csv", index=False)

# ======================= source 2: usage system (Dec 2024) =======================
absent = rng.choice(N, 25, replace=False)                                # SIMs with no usage record
use = pd.DataFrame({
    "customer_id": base.customer_id,
    "data_gb": base.data_gb,
    "voice_minutes": np.round(rng.gamma(2.2, 160, N) * (1 + base.tenure_months / 120)).astype(int),
    "recharge_count": np.where(base.plan_type == "Prepaid", rng.integers(1, 5, N), 0),
}).drop(index=absent)
glitch = rng.choice(use.index, 6, replace=False)
use.loc[glitch, "data_gb"] = [850.0, 912.5, 990.0, 875.2, 930.0, 999.9]  # meter glitches
low = rng.choice(use.index, 10, replace=False)
use.loc[low, "customer_id"] = use.loc[low, "customer_id"].str.lower()     # inconsistent key
use.to_csv(LAB / "usage_dec2024.csv", index=False)

# ======================= source 3: support tickets (Oct-Dec 2024) =======================
cats = ["Network", "Billing", "Recharge", "Data speed", "Other"]
rows = []
for cid, k in zip(base.customer_id, base.support_calls):
    for _ in range(int(k)):
        rows.append((cid, pd.Timestamp("2024-10-01") + pd.Timedelta(days=int(rng.integers(0, 92))),
                     rng.choice(cats, p=[.34, .22, .16, .2, .08])))
tk = pd.DataFrame(rows, columns=["customer_id", "ticket_date", "category"]).sort_values("ticket_date")
tk.insert(0, "ticket_id", [f"TK{50001 + i}" for i in range(len(tk))])
tk["ticket_date"] = tk.ticket_date.dt.strftime("%Y-%m-%d")
tk.to_csv(LAB / "support_tickets_q4_2024.csv", index=False)

print("crm", crm.shape, " usage", use.shape, " tickets", tk.shape)

# ======================= the reference cleaning pipeline =======================
raw = pd.read_csv(LAB / "crm_customers.csv", dtype=str, keep_default_na=False)
print("\nRAW CHECKS")
print("duplicates:", raw.duplicated().sum())
print("plan labels:", raw.plan_type.value_counts().to_dict())
print("gender labels:", raw.gender.value_counts().to_dict())
print("churned labels:", raw.churned.value_counts().to_dict())
print("empty cells:", {c: int((raw[c] == "").sum()) for c in raw.columns if (raw[c] == "").sum()})
print("charge with Rs symbol:", raw.monthly_charge.str.contains("₹").sum(), " with .00:", raw.monthly_charge.str.endswith(".00").sum())
print("dates with /:", raw.join_date.str.contains("/").sum())

c = raw.drop_duplicates().copy()
c["gender"] = c.gender.str.strip().str.lower().map({"m": "Male", "male": "Male", "f": "Female", "female": "Female"})
c["plan_type"] = c.plan_type.str.lower().str.replace("-", "", regex=False).map({"prepaid": "Prepaid", "postpaid": "Postpaid"})
c["churned"] = c.churned.str.title()
c["monthly_charge"] = pd.to_numeric(c.monthly_charge.str.replace("₹", "", regex=False))
c["join_date"] = pd.to_datetime(c.join_date, format="mixed", dayfirst=True)
c["age"] = pd.to_numeric(c.age.replace("", np.nan))
c["satisfaction"] = pd.to_numeric(c.satisfaction.replace("", np.nan))
print("\nafter dedupe:", len(c), " charge > 999:", int((c.monthly_charge > 999).sum()),
      " ages outside 18-100:", int(((c.age < 18) | (c.age > 100)).sum()), " age missing:", int(c.age.isna().sum()),
      " satisfaction missing:", int(c.satisfaction.isna().sum()))
print("age missing by plan %:", (c.groupby("plan_type").age.apply(lambda s: s.isna().mean() * 100)).round(1).to_dict())
print("churn % when satisfaction missing vs present:",
      round(c[c.satisfaction.isna()].churned.eq("Yes").mean() * 100, 1), round(c[c.satisfaction.notna()].churned.eq("Yes").mean() * 100, 1))

u = pd.read_csv(LAB / "usage_dec2024.csv")
u["customer_id"] = u.customer_id.str.upper()
dq = u.data_gb.quantile([.25, .5, .75])
q1, q3 = dq[.25], dq[.75]
iqr = q3 - q1
lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
z = (u.data_gb - u.data_gb.mean()) / u.data_gb.std()
print(f"\ndata_gb raw: mean {u.data_gb.mean():.2f} median {dq[.5]:.2f} std {u.data_gb.std():.2f} Q1 {q1:.2f} Q3 {q3:.2f} IQR {iqr:.2f} fences [{lo_f:.2f}, {hi_f:.2f}]")
print("  flagged by IQR:", int(((u.data_gb < lo_f) | (u.data_gb > hi_f)).sum()), " by |z|>3:", int((z.abs() > 3).sum()),
      " values > 100 GB:", int((u.data_gb > 100).sum()))
ok = u.data_gb <= 100
print(f"  without glitches: mean {u.data_gb[ok].mean():.2f} std {u.data_gb[ok].std():.2f} max {u.data_gb[ok].max():.1f}")
q1b, q3b = u.data_gb[ok].quantile([.25, .75])
print(f"  IQR fences on clean values: [{q1b - 1.5 * (q3b - q1b):.2f}, {q3b + 1.5 * (q3b - q1b):.2f}], real heavy users above upper fence:",
      int((u.data_gb[ok] > q3b + 1.5 * (q3b - q1b)).sum()))
vm = u.voice_minutes
print(f"voice_minutes: mean {vm.mean():.1f} median {vm.median():.1f} skew {vm.skew():.2f}  log1p skew {np.log1p(vm).skew():.2f}")
u.loc[~ok, "data_gb"] = np.nan                       # glitches become missing, then imputed

t = pd.read_csv(LAB / "support_tickets_q4_2024.csv")
calls = t.groupby("customer_id").size().rename("support_calls")
print("ticket categories:", t.category.value_counts().to_dict())

m = c.merge(u, on="customer_id", how="left").merge(calls, on="customer_id", how="left")
m["support_calls"] = m.support_calls.fillna(0).astype(int)
print("\nmerged:", m.shape, " no usage record:", int(m.voice_minutes.isna().sum()))

# outliers / invalid values -> fix or set missing
m.loc[m.monthly_charge > 999, "monthly_charge"] = m.loc[m.monthly_charge > 999, "monthly_charge"] / 100
m.loc[(m.age < 18) | (m.age > 100), "age"] = np.nan
# imputation comparisons on age
age_obs = m.age.dropna()
for name, filled in [("drop rows", age_obs),
                     ("mean", m.age.fillna(age_obs.mean())),
                     ("median", m.age.fillna(age_obs.median())),
                     ("group median", m.age.fillna(m.groupby("plan_type").age.transform("median")))]:
    print(f"age {name:12}: n {len(filled)} mean {filled.mean():.2f} std {filled.std():.2f}")
print("age median by plan:", m.groupby("plan_type").age.median().to_dict(), " overall mean", round(age_obs.mean(), 2), "median", age_obs.median())
m["age_was_missing"] = m.age.isna().astype(int)
m["age"] = m.age.fillna(m.groupby("plan_type").age.transform("median"))
m["satisfaction_missing"] = m.satisfaction.isna().astype(int)
m["satisfaction"] = m.satisfaction.fillna(m.satisfaction.median())
for col in ["data_gb", "voice_minutes", "recharge_count"]:
    m[col] = m[col].fillna(m[col].median())
m["tenure_months"] = np.floor((pd.Timestamp("2025-01-01") - m.join_date).dt.days / 30.4).astype(int).clip(1, 72)
print("tenure rebuilt matches Unit 1 for", round((m.set_index("customer_id").tenure_months
      .reindex(base.customer_id).values == base.tenure_months.values).mean() * 100, 1), "% of customers")

# transformations
m["churn_flag"] = (m.churned == "Yes").astype(int)
m["log_voice"] = np.log1p(m.voice_minutes)
m["charge_minmax"] = (m.monthly_charge - m.monthly_charge.min()) / (m.monthly_charge.max() - m.monthly_charge.min())
m["charge_z"] = (m.monthly_charge - m.monthly_charge.mean()) / m.monthly_charge.std()
m["tenure_band"] = pd.cut(m.tenure_months, [0, 6, 12, 24, 48, 72], labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
m["tier_ordinal"] = m.city_tier.map({"Tier-1": 1, "Tier-2": 2, "Tier-3": 3})
m["prepaid"] = (m.plan_type == "Prepaid").astype(int)
m["autopay_yes"] = (m.autopay == "Yes").astype(int)
m["male"] = (m.gender == "Male").astype(int)
m["gb_per_100rs"] = (m.data_gb / m.monthly_charge * 100).round(2)
print(f"\ncharge: min {m.monthly_charge.min()} max {m.monthly_charge.max()} mean {m.monthly_charge.mean():.2f} std {m.monthly_charge.std():.2f}")
ex = m.iloc[0]
print("example row:", ex.customer_id, ex.monthly_charge, round(ex.charge_minmax, 4), round(ex.charge_z, 4))

out = m[["customer_id", "gender", "age", "age_was_missing", "city_tier", "tier_ordinal", "plan_type", "prepaid", "join_date",
         "tenure_months", "tenure_band", "monthly_charge", "charge_minmax", "charge_z", "data_gb", "gb_per_100rs", "voice_minutes",
         "log_voice", "recharge_count", "autopay", "autopay_yes", "support_calls", "satisfaction", "satisfaction_missing", "male",
         "churned", "churn_flag"]].copy()
out["join_date"] = out.join_date.dt.strftime("%Y-%m-%d")
out = out.round(4)
out.to_csv(LAB / "swifttel_prepared_reference.csv", index=False)

# ======================= feature selection numbers =======================
num = ["tenure_months", "monthly_charge", "data_gb", "voice_minutes", "support_calls", "age", "satisfaction", "recharge_count",
       "prepaid", "autopay_yes", "tier_ordinal", "male", "satisfaction_missing"]
r = m[num + ["churn_flag"]].corr()["churn_flag"].drop("churn_flag").sort_values(key=abs, ascending=False)
print("\ncorrelation with churn:", r.round(3).to_dict())
print("r(data_gb, monthly_charge) =", round(m.data_gb.corr(m.monthly_charge), 3))
for col in ["plan_type", "autopay", "city_tier", "gender"]:
    tab = pd.crosstab(m[col], m.churned)
    chi2, p, dof, _ = chi2_contingency(tab, correction=False)
    print(f"chi-square {col:10}: chi2 {chi2:.2f} dof {dof} p {p:.4f}  table {tab.to_dict('index')}")
mi = pd.Series(mutual_info_classif(m[num], m.churn_flag, discrete_features=[c in ("support_calls", "prepaid", "autopay_yes", "tier_ordinal", "male", "satisfaction_missing", "recharge_count") for c in num], random_state=0), index=num).sort_values(ascending=False)
print("mutual information:", mi.round(4).to_dict())
Xs = StandardScaler().fit_transform(m[num])
rfe = RFE(LogisticRegression(max_iter=5000), n_features_to_select=5).fit(Xs, m.churn_flag)
print("RFE keeps:", [n for n, k in zip(num, rfe.support_) if k], " ranking:", dict(zip(num, rfe.ranking_.tolist())))
l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.02).fit(Xs, m.churn_flag)
print("L1 (C=0.02) non-zero:", {n: round(w, 3) for n, w in zip(num, l1.coef_[0]) if abs(w) > 1e-9})
print("variance of male flag:", round(m.male.var(), 3), " of satisfaction_missing:", round(m.satisfaction_missing.var(), 4))
desc = m[["age", "tenure_months", "monthly_charge", "data_gb", "voice_minutes", "support_calls"]].describe().round(2)
print("\n", desc)
print("skewness:", m[["age", "tenure_months", "monthly_charge", "data_gb", "voice_minutes", "support_calls"]].skew().round(2).to_dict())

# ======================= charts =======================
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 130})
BLUE, ORANGE, GREY, GREEN, RED = "#2a6fdb", "#e8833a", "#9aa4b2", "#1f9d6b", "#d64545"

def save(fig, name):
    fig.tight_layout(); fig.savefig(IMG / name, facecolor="white"); plt.close(fig)

mm = c.merge(pd.read_csv(LAB / "usage_dec2024.csv").assign(customer_id=lambda d: d.customer_id.str.upper()), on="customer_id", how="left")
miss = mm.isna().sum(); miss = miss[miss > 0].sort_values()
fig, ax = plt.subplots(figsize=(6.6, 2.8))
ax.barh(miss.index, miss.values, color=ORANGE)
for yy, v in enumerate(miss.values):
    ax.text(v + 1, yy, str(v), va="center")
ax.set_title("Missing values after merging CRM + usage"); ax.set_xlabel("empty cells")
save(fig, "missing_after_merge.png")

raw_gb = pd.read_csv(LAB / "usage_dec2024.csv").data_gb
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
axes[0].boxplot(raw_gb, vert=False, widths=.5, flierprops=dict(marker="o", markerfacecolor=RED, markersize=6))
axes[0].set_title("data_gb as received"); axes[0].set_xlabel("GB per month"); axes[0].set_yticks([])
axes[0].annotate("6 meter glitches\n(850–1000 GB)", xy=(900, 1), xytext=(520, 1.32), arrowprops=dict(arrowstyle="->", color=GREY), fontsize=9)
axes[1].boxplot(raw_gb[raw_gb <= 100], vert=False, widths=.5, flierprops=dict(marker="o", markerfacecolor=ORANGE, markersize=5))
axes[1].set_title("after removing the glitches"); axes[1].set_xlabel("GB per month"); axes[1].set_yticks([])
axes[1].annotate("real heavy users:\nkeep them", xy=(84, 1), xytext=(50, 1.32), arrowprops=dict(arrowstyle="->", color=GREY), fontsize=9)
save(fig, "outliers_boxplot.png")

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.1))
axes[0].hist(vm, bins=40, color=BLUE); axes[0].set_title(f"voice_minutes (skew {vm.skew():.2f})"); axes[0].set_xlabel("minutes")
axes[1].hist(np.log1p(vm), bins=40, color=GREEN); axes[1].set_title(f"log(1 + minutes) (skew {np.log1p(vm).skew():.2f})"); axes[1].set_xlabel("log scale")
save(fig, "log_transform.png")

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.1), sharey=True)
bins = np.arange(15, 72, 2)
axes[0].hist(m.age.where(m.age_was_missing == 0).dropna(), bins=bins, color=GREY, label="observed")
axes[0].hist([age_obs.mean()] * int(m.age_was_missing.sum()), bins=bins, color=RED, label="filled with mean")
axes[0].set_title("Mean imputation: a spike"); axes[0].legend(frameon=False, fontsize=9); axes[0].set_xlabel("age")
axes[1].hist(m.age.where(m.age_was_missing == 0).dropna(), bins=bins, color=GREY, label="observed")
axes[1].hist(m.age[m.age_was_missing == 1], bins=bins, color=GREEN, label="filled with plan median")
axes[1].set_title("Group median: smaller, split spikes"); axes[1].legend(frameon=False, fontsize=9); axes[1].set_xlabel("age")
save(fig, "imputation_compare.png")

cols = ["tenure_months", "monthly_charge", "data_gb", "voice_minutes", "support_calls", "satisfaction", "age", "churn_flag"]
cm = m[cols].corr()
fig, ax = plt.subplots(figsize=(7.2, 5.6))
im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)), [x.replace("_", "\n") for x in cols], fontsize=8.5); ax.set_yticks(range(len(cols)), cols, fontsize=9)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(cm.iloc[i, j]) > .55 else "black")
ax.spines[:].set_visible(False); fig.colorbar(im, shrink=.8); ax.set_title("Correlation heatmap")
save(fig, "correlation_heatmap.png")

fig, ax = plt.subplots(figsize=(6.4, 3.6))
for k, col in [(0, BLUE), (1, ORANGE)]:
    d = m[m.churn_flag == k]
    ax.scatter(d.monthly_charge + rng.normal(0, 8, len(d)), d.data_gb, s=8, alpha=.45, color=col, label="churned" if k else "stayed")
ax.set_xlabel("monthly charge (₹)"); ax.set_ylabel("data used (GB)")
ax.set_title(f"Two features carrying the same signal (r = {m.data_gb.corr(m.monthly_charge):.2f})"); ax.legend(frameon=False)
save(fig, "collinear_scatter.png")

rank = mi.sort_values()
fig, ax = plt.subplots(figsize=(6.6, 4))
ax.barh(rank.index, rank.values, color=[GREEN if v >= rank.values[-5] else GREY for v in rank.values])
ax.set_title("Mutual information with churn"); ax.set_xlabel("higher = more informative")
save(fig, "feature_ranking.png")

tab = m.groupby(["city_tier", "plan_type"]).churn_flag.mean().unstack() * 100
fig, ax = plt.subplots(figsize=(6.4, 3.2))
xx = np.arange(len(tab))
ax.bar(xx - .2, tab["Postpaid"], .4, color=BLUE, label="Postpaid"); ax.bar(xx + .2, tab["Prepaid"], .4, color=ORANGE, label="Prepaid")
for i, (a, b) in enumerate(zip(tab["Postpaid"], tab["Prepaid"])):
    ax.text(i - .2, a + .6, f"{a:.0f}%", ha="center", fontsize=9); ax.text(i + .2, b + .6, f"{b:.0f}%", ha="center", fontsize=9)
ax.set_xticks(xx, tab.index); ax.set_ylabel("% churned"); ax.legend(frameon=False); ax.set_title("Churn by city tier and plan")
save(fig, "churn_tier_plan.png")
print("churn % tier x plan:", tab.round(1).to_dict())
print("charts written")
