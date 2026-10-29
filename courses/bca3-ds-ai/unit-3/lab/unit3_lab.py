"""
Unit 3 Lab · Predictive Modeling Techniques
Regression (used-car prices) and classification (SwiftTel churn) with the model-building process.

Files : used_cars.csv            (1,200 used-car listings; target: price_lakh)
        swifttel_customers.csv   (from Unit 1; target: churned)
Needs : pandas, numpy, scikit-learn, matplotlib   ->  pip install pandas numpy scikit-learn matplotlib
Run   : python unit3_lab.py      (both CSV files in the same folder)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
pd.set_option("display.width", 120)

# ============================================================
# PART 1 · REGRESSION: SPLIT THE CAR DATA
# ============================================================
# Model-building step 1-3: define the target, choose features, split BEFORE anything is learned.
cars = pd.read_csv(HERE / "used_cars.csv")
print(cars.head(3).to_string())
train, test = train_test_split(cars, test_size=0.3, random_state=42)
print("train", train.shape, " test", test.shape)

# ============================================================
# PART 2 · SIMPLE LINEAR REGRESSION (ONE FEATURE)
# ============================================================
simple = LinearRegression().fit(train[["age_years"]], train["price_lakh"])
pred = simple.predict(test[["age_years"]])
print(f"\nprice = {simple.intercept_:.3f} + ({simple.coef_[0]:.3f}) x age")
print(f"test R2 {r2_score(test['price_lakh'], pred):.3f}   RMSE {mean_squared_error(test['price_lakh'], pred) ** 0.5:.3f}   "
      f"MAE {mean_absolute_error(test['price_lakh'], pred):.3f}")

# ============================================================
# PART 3 · MULTIPLE LINEAR REGRESSION (MANY FEATURES + DUMMIES)
# ============================================================
def features(d):
    X = pd.DataFrame({"age_years": d["age_years"], "km_thousand": d["km_driven"] / 1000, "owners": d["owners"]})
    X = X.join(pd.get_dummies(d["brand_tier"], prefix="tier", dtype=int).drop(columns="tier_Budget", errors="ignore"))   # Budget = baseline
    X = X.join(pd.get_dummies(d["fuel"], prefix="fuel", dtype=int).drop(columns="fuel_Petrol", errors="ignore"))   # Petrol = baseline
    X["automatic"] = (d["transmission"] == "Automatic").astype(int)
    return X

Xtr, Xte = features(train), features(test)
multi = LinearRegression().fit(Xtr, train["price_lakh"])
pred = multi.predict(Xte)
print(f"\nintercept {multi.intercept_:.3f}")
for name, w in zip(Xtr.columns, multi.coef_):
    print(f"  {name:14} {w:+.4f}")
r2_tr = r2_score(train["price_lakh"], multi.predict(Xtr))
n, k = Xtr.shape
print(f"train R2 {r2_tr:.3f}   adjusted R2 {1 - (1 - r2_tr) * (n - 1) / (n - k - 1):.3f}")
print(f"test  R2 {r2_score(test['price_lakh'], pred):.3f}   RMSE {mean_squared_error(test['price_lakh'], pred) ** 0.5:.3f}   "
      f"MAE {mean_absolute_error(test['price_lakh'], pred):.3f}")
one = test.iloc[[0]]
print("example:", one[["brand_tier", "age_years", "km_driven", "fuel", "transmission", "owners", "price_lakh"]].to_dict("records")[0],
      " predicted", round(float(multi.predict(features(one).reindex(columns=Xtr.columns, fill_value=0))[0]), 2))

# ============================================================
# PART 4 · CLASSIFICATION: PREPARE AND SPLIT THE CHURN DATA
# ============================================================
df = pd.read_csv(HERE / "swifttel_customers.csv")
X = pd.DataFrame({"tenure_months": df["tenure_months"], "monthly_charge": df["monthly_charge"],
                  "support_calls": df["support_calls"], "prepaid": (df["plan_type"] == "Prepaid").astype(int),
                  "autopay": (df["autopay"] == "Yes").astype(int), "satisfaction": df["satisfaction"].astype(float)})
y = (df["churned"] == "Yes").astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
median_sat = X_train["satisfaction"].median()                   # learned from TRAINING data only
X_train = X_train.fillna({"satisfaction": median_sat})
X_test = X_test.fillna({"satisfaction": median_sat})
scaler = StandardScaler().fit(X_train)                           # also fitted on training data only
Z_train, Z_test = scaler.transform(X_train), scaler.transform(X_test)
print(f"\ntrain {len(X_train)} customers ({y_train.mean():.1%} churn)   test {len(X_test)} ({y_test.mean():.1%} churn)")

# ============================================================
# PART 5 · FOUR CLASSIFIERS
# ============================================================
models = {
    "Logistic regression": (LogisticRegression(max_iter=5000), X_train, X_test),
    "Decision tree (depth 3)": (DecisionTreeClassifier(max_depth=3, random_state=0), X_train, X_test),
    "KNN (k = 15, scaled)": (KNeighborsClassifier(n_neighbors=15), Z_train, Z_test),
    "Naive Bayes (Gaussian)": (GaussianNB(), X_train, X_test),
}
for name, (model, A, B) in models.items():
    model.fit(A, y_train)
    p = model.predict(B)
    print(f"{name:25} accuracy {accuracy_score(y_test, p):.3f}  precision {precision_score(y_test, p, zero_division=0):.3f}  "
          f"recall {recall_score(y_test, p):.3f}")
print(f"{'Always predict No':25} accuracy {1 - y_test.mean():.3f}")

# ============================================================
# PART 6 · INTERPRET THE MODELS
# ============================================================
logit = models["Logistic regression"][0]
print("\nLogistic regression: coefficient and odds ratio e^b")
for name, b in zip(X.columns, logit.coef_[0]):
    print(f"  {name:15} b = {b:+.4f}   odds ratio = {np.exp(b):.3f}")
tree = models["Decision tree (depth 3)"][0]
print("\n" + export_text(tree, feature_names=list(X.columns)))
for k in (1, 5, 15, 51):
    knn = KNeighborsClassifier(n_neighbors=k).fit(Z_train, y_train)
    print(f"KNN k={k:2}: train accuracy {knn.score(Z_train, y_train):.3f}   test accuracy {knn.score(Z_test, y_test):.3f}")

# ============================================================
# PART 7 · SCORE A NEW CUSTOMER AND SAVE A CHART
# ============================================================
new = pd.DataFrame([{"tenure_months": 8, "monthly_charge": 399, "support_calls": 4, "prepaid": 1, "autopay": 0, "satisfaction": 2.0}])
for name, (model, A, _) in models.items():
    row = scaler.transform(new) if A is Z_train else new
    print(f"P(churn) for the new customer, {name:25}: {model.predict_proba(row)[0, 1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].scatter(test["price_lakh"], pred, s=8, alpha=.5)
axes[0].plot([0, 20], [0, 20], "--", color="orange"); axes[0].set_xlabel("actual"); axes[0].set_ylabel("predicted")
axes[0].set_title("Car prices: actual vs predicted")
axes[1].barh(list(X.columns), tree.feature_importances_); axes[1].set_title("Decision tree feature importance")
fig.tight_layout(); fig.savefig(HERE / "unit3_models.png", dpi=110)
print("\nSaved unit3_models.png")
