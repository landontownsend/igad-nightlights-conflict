"""
Models: do night lights help predict new outbreaks beyond past violence?

Reads data/processed/igad_locality_month.parquet and writes results/:
  metrics.csv          every model, target and test, with its scores
  lights_gain.csv      how much the lights add, with 95% intervals (bootstrap)
  coefficients.csv     logistic regression weights (odds ratios)
  importance.csv       gradient boosting: mean SHAP value of each predictor
  predictions.parquet  every test row's risk score, for figures and checks

Design (all decided in the proposal):
- Rows: at-risk locality-months only (12 quiet months behind them).
- Target: a new outbreak in this month or the next 2 (3-month window) or 5 (6-month).
  Predictors in a row use only earlier months, so this is a forecast.
- Train on 2015 to 2021 and test on 2022 to 2025. A training row counts only if its
  whole window ends by December 2021, so no outcome from the test years leaks in.
- Sudan holdout: train on the other six countries (2015 to 2021), test on Sudan (2022 to 2025).
- Three versions of each model: baseline (past violence + controls), baseline + lights,
  and lights + controls (no violence).

Run from the repository root:  python -m analysis.models
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data/processed/igad_locality_month.parquet")
OUT = Path("results")
SEED = 735

VIOLENCE = ["months_since_violence", "log_nearby_deaths_3m", "log_national_deaths_3m"]
CONTROLS = ["log_population", "log_prewar_light", "clear_nights"]
LIGHTS = ["light_change_3m", "light_change_6m", "sudden_drop", "months_declining"]
VERSIONS = {
    "baseline": VIOLENCE + CONTROLS,
    "baseline_plus_lights": VIOLENCE + CONTROLS + LIGHTS,
    "lights_only": LIGHTS + CONTROLS,
}
TRAIN_END, TEST_START = 2021, 2022


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def load():
    d = pd.read_parquet(DATA)
    d = d.sort_values(["locality_id", "year", "month"]).reset_index(drop=True)
    for col in d.columns:
        if str(d[col].dtype) == "Int64":
            d[col] = d[col].astype(float)
    # Counts and sizes are very skewed, so the models see their logs
    d["log_nearby_deaths_3m"] = np.log1p(d["nearby_deaths_3m"])
    d["log_national_deaths_3m"] = np.log1p(d["national_deaths_3m"])
    d["log_population"] = np.log(d["population"])
    d["log_prewar_light"] = np.log1p(d["prewar_light"])
    d["t"] = d["year"] * 12 + d["month"] - 1          # month counter

    last_t = d["t"].max()
    for k in (3, 6):
        # Any new outbreak in this month or the next k - 1 months
        future = sum(d.groupby("locality_id")["new_outbreak"].shift(-j) for j in range(k))
        d[f"y{k}"] = (future > 0).astype(float)
        d.loc[d["t"] + k - 1 > last_t, f"y{k}"] = np.nan   # window runs past the data
        d[f"window_end_year{k}"] = (d["t"] + k - 1) // 12
    return d[d["at_risk"] == 1].copy()


def split(d, k, sudan_holdout=False):
    y = f"y{k}"
    rows = d[d[y].notna()]
    train = rows[rows[f"window_end_year{k}"] <= TRAIN_END]
    test = rows[rows["year"] >= TEST_START]
    if sudan_holdout:
        train = train[train["country"] != "SDN"]
        test = test[test["country"] == "SDN"]
    return train, test


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

def fit_logit(train, features, y):
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=5000))
    model.fit(train[features], train[y])
    return model


GBM_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=7, max_depth=3,
                  min_child_samples=100, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1)


def fit_gbm(train, features, y):
    """Pick the number of trees on a time split inside the training years, then refit."""
    k = int(y[1:])
    early = train[train[f"window_end_year{k}"] <= TRAIN_END - 2]   # 2015 to 2019
    late = train[train["year"] >= TRAIN_END - 1]                    # 2020 to 2021
    probe = lgb.LGBMClassifier(n_estimators=2000, **GBM_PARAMS)
    probe.fit(early[features], early[y], eval_set=[(late[features], late[y])],
              eval_metric="average_precision",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    n_trees = max(probe.best_iteration_ or 100, 50)
    model = lgb.LGBMClassifier(n_estimators=n_trees, **GBM_PARAMS)
    model.fit(train[features], train[y])
    return model, n_trees


# -----------------------------------------------------------------------------
# Scores
# -----------------------------------------------------------------------------

def recall_at_top(test, score, y, share):
    """Each month, flag the riskiest `share` of at-risk localities. Share of outbreaks caught."""
    df = pd.DataFrame({"t": test["t"].to_numpy(), "s": score, "y": test[y].to_numpy()})
    df["rank"] = df.groupby("t")["s"].rank(ascending=False, method="first")
    df["n"] = df.groupby("t")["s"].transform("size")
    flagged = df["rank"] <= np.ceil(share * df["n"])
    return df.loc[flagged, "y"].sum() / df["y"].sum()


def scores(test, score, y):
    return {"pr_auc": average_precision_score(test[y], score),
            "roc_auc": roc_auc_score(test[y], score),
            "recall_top5": recall_at_top(test, score, y, 0.05),
            "recall_top10": recall_at_top(test, score, y, 0.10),
            "base_rate": test[y].mean(), "positives": int(test[y].sum()), "rows": len(test)}


def bootstrap_gain(test, s_base, s_lights, y, n=1000):
    """Resample whole localities, so a locality's months stay together."""
    rng = np.random.default_rng(SEED)
    ids = test["locality_id"].to_numpy()
    unique = np.unique(ids)
    index = {u: np.flatnonzero(ids == u) for u in unique}
    yv = test[y].to_numpy()
    gains = []
    for _ in range(n):
        rows = np.concatenate([index[u] for u in rng.choice(unique, len(unique))])
        if yv[rows].sum() == 0:
            continue
        gains.append(average_precision_score(yv[rows], s_lights[rows])
                     - average_precision_score(yv[rows], s_base[rows]))
    gains = np.array(gains)
    return {"gain": average_precision_score(yv, s_lights) - average_precision_score(yv, s_base),
            "low": np.percentile(gains, 2.5), "high": np.percentile(gains, 97.5),
            "share_above_zero": (gains > 0).mean()}


# -----------------------------------------------------------------------------
# Run everything
# -----------------------------------------------------------------------------

def main():
    OUT.mkdir(exist_ok=True)
    d = load()
    metrics, gains, coefs, importance, preds = [], [], [], [], []

    for test_name, holdout in [("2022-2025", False), ("Sudan 2022-2025", True)]:
        for k in (3, 6):
            y = f"y{k}"
            train, test = split(d, k, holdout)
            print(f"\n== {test_name}, {k}-month window: train {len(train):,} rows "
                  f"({int(train[y].sum())} positive), test {len(test):,} ({int(test[y].sum())} positive)")
            scored = {}
            for version, features in VERSIONS.items():
                logit = fit_logit(train, features, y)
                gbm, n_trees = fit_gbm(train, features, y)
                for model_name, model in [("logistic", logit), ("gradient_boosting", gbm)]:
                    s = model.predict_proba(test[features])[:, 1]
                    scored[(model_name, version)] = s
                    m = scores(test, s, y)
                    m.update(test=test_name, window=k, model=model_name, version=version,
                             trees=n_trees if model_name == "gradient_boosting" else None)
                    metrics.append(m)
                    print(f"   {model_name:18} {version:21} PR-AUC {m['pr_auc']:.4f}  ROC-AUC {m['roc_auc']:.3f}"
                          f"  top10 recall {m['recall_top10']:.2f}")
                    preds.append(pd.DataFrame({"test": test_name, "window": k, "model": model_name,
                                               "version": version, "locality_id": test["locality_id"].to_numpy(),
                                               "country": test["country"].to_numpy(),
                                               "locality_name": test["locality_name"].to_numpy(),
                                               "year": test["year"].to_numpy(), "month": test["month"].to_numpy(),
                                               "y": test[y].to_numpy(), "risk": s}))
                if not holdout and version == "baseline_plus_lights":
                    lr = logit[-1]
                    coefs.append(pd.DataFrame({"window": k, "feature": features, "coef_per_sd": lr.coef_[0],
                                               "odds_ratio_per_sd": np.exp(lr.coef_[0])}))
                    sv = shap.TreeExplainer(gbm).shap_values(test[features])
                    sv = sv[1] if isinstance(sv, list) else sv
                    importance.append(pd.DataFrame({"window": k, "feature": features,
                                                    "mean_abs_shap": np.abs(sv).mean(axis=0)}))
            for model_name in ("logistic", "gradient_boosting"):
                g = bootstrap_gain(test, scored[(model_name, "baseline")],
                                   scored[(model_name, "baseline_plus_lights")], y)
                g.update(test=test_name, window=k, model=model_name)
                gains.append(g)
                print(f"   lights gain, {model_name}: {g['gain']:+.4f} PR-AUC "
                      f"(95% interval {g['low']:+.4f} to {g['high']:+.4f}; above zero in {g['share_above_zero']:.0%})")

    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(gains).to_csv(OUT / "lights_gain.csv", index=False)
    pd.concat(coefs).to_csv(OUT / "coefficients.csv", index=False)
    pd.concat(importance).to_csv(OUT / "importance.csv", index=False)
    pd.concat(preds).to_parquet(OUT / "predictions.parquet", index=False)
    print("\nSaved results/")


if __name__ == "__main__":
    main()
