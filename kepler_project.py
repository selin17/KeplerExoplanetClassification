"""
Data Mining Term Project - Kepler Exoplanet Candidate Classification
Task : CONFIRMED planet vs FALSE POSITIVE (binary classification)
Data : NASA Exoplanet Archive, Kepler Objects of Interest (KOI) Q1-Q17 DR25 table

Usage:  python kepler_project.py path/to/q1_q17_dr25_koi.csv
Outputs are written to ./outputs
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score, roc_curve,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
CSV = sys.argv[1] if len(sys.argv) > 1 else "q1_q17_dr25_koi_2026.10.30_06.37.09.csv"

# ----------------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------------
raw = pd.read_csv(CSV, comment="#")
print(f"Raw table: {raw.shape[0]} rows x {raw.shape[1]} columns")
print(raw["koi_disposition"].value_counts(), "\n")

labeled = raw[raw["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
candidates = raw[raw["koi_disposition"] == "CANDIDATE"].copy()   # kept aside, predicted at the end
y = (labeled["koi_disposition"] == "CONFIRMED").astype(int)       # 1 = planet, 0 = false positive
groups = labeled["kepid"]                                          # same star -> same fold (no leakage)
print(f"Labeled: {len(labeled)} | CONFIRMED share: {y.mean():.3f} | unique stars: {groups.nunique()}")

# ----------------------------------------------------------------------------
# 2. FEATURE SELECTION (leakage control)
#    - identifiers / names: not physical features
#    - koi_pdisposition, koi_score: output of Kepler's own vetting/classifier  -> leakage
#    - koi_fpflag_*: output of the false-positive vetting process              -> leakage
#    - koi_teq_err1/err2: completely empty
# ----------------------------------------------------------------------------
ID_COLS = ["kepid", "kepoi_name", "kepler_name", "koi_disposition"]
LEAK_SCORE = ["koi_pdisposition", "koi_score"]
LEAK_FLAGS = [c for c in raw.columns if c.startswith("koi_fpflag")]
EMPTY = [c for c in raw.columns if raw[c].isna().all()]
print("Empty columns dropped:", EMPTY)

X_full = labeled.drop(columns=ID_COLS + EMPTY)
X = X_full.drop(columns=LEAK_SCORE + LEAK_FLAGS)                   # MAIN feature set
CAT = ["koi_tce_delivname"]
NUM = [c for c in X.columns if c not in CAT]
print(f"Main feature set: {len(NUM)} numeric + {len(CAT)} categorical\n")


# ----------------------------------------------------------------------------
# 3. EDA FIGURES (class balance, missing values, outliers)
# ----------------------------------------------------------------------------
class IQRClipper(BaseEstimator, TransformerMixin):
    """Winsorise: clip every column to [Q1 - k*IQR, Q3 + k*IQR] learned on the training data only."""
    def __init__(self, k=3.0):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lo_, self.hi_ = q1 - self.k * iqr, q3 + self.k * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, names=None):
        return names


fig, ax = plt.subplots(1, 3, figsize=(16, 4))
raw["koi_disposition"].value_counts().plot.bar(ax=ax[0], color=["#c0504d", "#4f81bd", "#9bbb59"])
ax[0].set_title("Class distribution (raw table)")
ax[0].tick_params(axis="x", rotation=15)
miss = X.isna().mean().sort_values(ascending=False).head(10) * 100
miss.plot.barh(ax=ax[1], color="#4f81bd"); ax[1].invert_yaxis()
ax[1].set_title("Missing values % (top 10, main features)")
show = ["koi_period", "koi_depth", "koi_prad", "koi_model_snr"]
before = X[show].copy()
after = pd.DataFrame(IQRClipper(3.0).fit(before).transform(before), columns=show)
pos = np.arange(len(show))
ax[2].boxplot([np.log10(before[c].clip(lower=1e-3).dropna()) for c in show], positions=pos - 0.18,
              widths=0.3, patch_artist=True, boxprops=dict(facecolor="#e6b8b7"), showfliers=True,
              flierprops=dict(markersize=2))
ax[2].boxplot([np.log10(after[c].clip(lower=1e-3).dropna()) for c in show], positions=pos + 0.18,
              widths=0.3, patch_artist=True, boxprops=dict(facecolor="#b8cce4"), showfliers=True,
              flierprops=dict(markersize=2))
ax[2].set_xticks(pos); ax[2].set_xticklabels(show, fontsize=8)
ax[2].set_title("log10 scale: before (red) vs after IQR clipping (blue)")
plt.tight_layout(); plt.savefig(OUT / "01_eda.png", dpi=150); plt.close()

# ----------------------------------------------------------------------------
# 4. PREPROCESSING PIPELINE  (fit only on training folds -> no leakage)
#    numeric : median impute -> IQR clip (k=3) -> Yeo-Johnson + standardise
#    category: most-frequent impute -> one-hot
# ----------------------------------------------------------------------------
def make_pre():
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("clip", IQRClipper(3.0)),
                    ("scale", PowerTransformer(method="yeo-johnson", standardize=True))])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])


# ----------------------------------------------------------------------------
# 5. TRAIN / TEST SPLIT (80/20, stratified, grouped by star)
# ----------------------------------------------------------------------------
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
tr_idx, te_idx = next(sgkf.split(X, y, groups))
X_tr, X_te, y_tr, y_te = X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx]
g_tr = groups.iloc[tr_idx]
print(f"Train: {len(X_tr)}  Test: {len(X_te)}  (test CONFIRMED share {y_te.mean():.3f})\n")

# ----------------------------------------------------------------------------
# 6. MODELS + GRID SEARCH (5-fold stratified-group CV on the training set)
# ----------------------------------------------------------------------------
MODELS = {
    "Decision Tree": (DecisionTreeClassifier(random_state=SEED),
                      {"clf__max_depth": [4, 8, 12, None], "clf__min_samples_leaf": [1, 5, 20]}),
    "Random Forest": (RandomForestClassifier(random_state=SEED, n_jobs=-1),
                      {"clf__n_estimators": [200, 400], "clf__max_depth": [10, None],
                       "clf__min_samples_leaf": [1, 3]}),
    "SVM (RBF)": (SVC(kernel="rbf", probability=True, random_state=SEED),
                  {"clf__C": [1, 10], "clf__gamma": ["scale", 0.01]}),
    "MLP": (MLPClassifier(max_iter=600, early_stopping=True, random_state=SEED),
            {"clf__hidden_layer_sizes": [(64,), (64, 32)], "clf__alpha": [1e-4, 1e-2]}),
    "Naive Bayes": (GaussianNB(), {"clf__var_smoothing": [1e-9, 1e-7, 1e-5]}),
    "Logistic Regression": (LogisticRegression(max_iter=2000), {"clf__C": [0.1, 1, 10]}),
}


def metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "ROC-AUC": roc_auc_score(y_true, y_prob),
            "Specificity": tn / (tn + fp),
            "MCC": matthews_corrcoef(y_true, y_pred),
            "Kappa": cohen_kappa_score(y_true, y_pred),
            "Balanced Acc": balanced_accuracy_score(y_true, y_pred)}


rows, fitted, roc_data, best_params = [], {}, {}, {}
for name, (clf, grid) in MODELS.items():
    pipe = Pipeline([("pre", make_pre()), ("clf", clf)])
    gs = GridSearchCV(pipe, grid, scoring="roc_auc",
                      cv=StratifiedGroupKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    gs.fit(X_tr, y_tr, groups=g_tr)
    prob = gs.predict_proba(X_te)[:, 1]
    pred = gs.predict(X_te)
    res = metrics(y_te, pred, prob)
    res.update({"Model": name, "CV AUC (train)": gs.best_score_})
    rows.append(res); fitted[name] = gs.best_estimator_; best_params[name] = gs.best_params_
    fpr, tpr, _ = roc_curve(y_te, prob); roc_data[name] = (fpr, tpr, res["ROC-AUC"])
    print(f"{name:20s} AUC={res['ROC-AUC']:.3f}  F1={res['F1']:.3f}  best={gs.best_params_}")

table = pd.DataFrame(rows).set_index("Model")
cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Specificity", "MCC", "Kappa",
        "Balanced Acc", "CV AUC (train)"]
table = table[cols].sort_values("ROC-AUC", ascending=False).round(4)
table.to_csv(OUT / "02_model_comparison.csv")
pd.Series({k: str(v) for k, v in best_params.items()}).to_csv(OUT / "02b_best_params.csv", header=["params"])
print("\n", table.to_string(), "\n")

# single ROC plot
plt.figure(figsize=(6.5, 5.5))
for n, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{n} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=0.8)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves - hold-out test set"); plt.legend(fontsize=8, loc="lower right")
plt.tight_layout(); plt.savefig(OUT / "03_roc_all_models.png", dpi=150); plt.close()

# confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, (n, m) in zip(axes.ravel(), fitted.items()):
    ConfusionMatrixDisplay.from_predictions(y_te, m.predict(X_te), ax=ax, colorbar=False,
                                            display_labels=["False pos.", "Planet"], cmap="Blues")
    ax.set_title(n)
plt.tight_layout(); plt.savefig(OUT / "04_confusion_matrices.png", dpi=150); plt.close()

best_name = table.index[0]
print("Best model (by test ROC-AUC):", best_name)

# ----------------------------------------------------------------------------
# 7. FEATURE IMPORTANCE (permutation, on test set, best tree-based model)
# ----------------------------------------------------------------------------
imp_model = fitted["Random Forest"]
pi = permutation_importance(imp_model, X_te, y_te, scoring="roc_auc", n_repeats=8,
                            random_state=SEED, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
imp.to_csv(OUT / "05_feature_importance.csv", header=["AUC drop when shuffled"])
imp.head(12)[::-1].plot.barh(figsize=(7, 5), color="#4f81bd")
plt.title("Random Forest - permutation importance (test set)"); plt.xlabel("Drop in ROC-AUC")
plt.tight_layout(); plt.savefig(OUT / "05_feature_importance.png", dpi=150); plt.close()
print("\nTop features:\n", imp.head(8).round(4).to_string(), "\n")

# ----------------------------------------------------------------------------
# 8. EXPERIMENT A - class balancing (none / class_weight / SMOTE), Random Forest
# ----------------------------------------------------------------------------
cvk = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
rf_kw = dict(n_estimators=300, random_state=SEED, n_jobs=-1)
variants = {
    "No balancing": Pipeline([("pre", make_pre()), ("clf", RandomForestClassifier(**rf_kw))]),
    "class_weight=balanced": Pipeline([("pre", make_pre()),
                                       ("clf", RandomForestClassifier(class_weight="balanced", **rf_kw))]),
    "SMOTE (train folds only)": ImbPipeline([("pre", make_pre()), ("smote", SMOTE(random_state=SEED)),
                                             ("clf", RandomForestClassifier(**rf_kw))]),
}
bal_rows = []
for n, p in variants.items():
    r = cross_validate(p, X, y, cv=cvk, groups=groups, scoring=["accuracy", "recall", "f1", "roc_auc"])
    bal_rows.append({"Variant": n, **{k.replace("test_", ""): r[k].mean() for k in r if k.startswith("test_")}})
bal = pd.DataFrame(bal_rows).set_index("Variant").round(4)
bal.to_csv(OUT / "06_balancing_experiment.csv")
print("Experiment A - balancing (5-fold grouped CV):\n", bal.to_string(), "\n")

# ----------------------------------------------------------------------------
# 9. EXPERIMENT B - data leakage check (Random Forest, grouped 5-fold CV)
# ----------------------------------------------------------------------------
def cv_rf(Xs, cat, num):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("clip", IQRClipper(3.0)),
                          ("scale", PowerTransformer())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    p = Pipeline([("pre", pre), ("clf", RandomForestClassifier(**rf_kw))])
    r = cross_validate(p, Xs, y, cv=cvk, groups=groups, scoring=["accuracy", "f1", "roc_auc"])
    return {"Accuracy": r["test_accuracy"].mean(), "F1": r["test_f1"].mean(), "ROC-AUC": r["test_roc_auc"].mean()}


sets = {
    "A) + koi_score + fpflags (LEAKY)": X_full,
    "B) + fpflags, no score (LEAKY)": X_full.drop(columns=LEAK_SCORE),
    "C) clean features (MAIN)": X,
    "D) clean, no stellar-temp errors": X.drop(columns=["koi_steff_err1", "koi_steff_err2"]),
}
leak_rows = []
for n, Xs in sets.items():
    cat = [c for c in CAT + ["koi_pdisposition"] if c in Xs.columns]   # text columns -> one-hot
    num = [c for c in Xs.columns if c not in cat]
    leak_rows.append({"Feature set": n, "n_features": Xs.shape[1], **cv_rf(Xs, cat, num)})
leak = pd.DataFrame(leak_rows).set_index("Feature set").round(4)
leak.to_csv(OUT / "07_leakage_experiment.csv")
print("Experiment B - leakage check:\n", leak.to_string(), "\n")

# ----------------------------------------------------------------------------
# 10. FINAL STEP - rank the CANDIDATE objects with the best model
# ----------------------------------------------------------------------------
final = fitted[best_name]
Xc = candidates.drop(columns=ID_COLS + EMPTY + LEAK_SCORE + LEAK_FLAGS)
cand = candidates[["kepoi_name", "kepid"]].copy()
cand["planet_probability"] = final.predict_proba(Xc)[:, 1]
cand = cand.sort_values("planet_probability", ascending=False)
cand.to_csv(OUT / "08_candidate_predictions.csv", index=False)
print(f"{len(cand)} CANDIDATE objects scored with {best_name}.")
print(f"  predicted planet (p>=0.5): {(cand.planet_probability >= 0.5).sum()}   "
      f"high confidence (p>=0.9): {(cand.planet_probability >= 0.9).sum()}")
print(cand.head(10).to_string(index=False))
print("\nDone. All outputs in ./outputs")
