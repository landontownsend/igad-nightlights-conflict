"""
Checks on the model results (run after analysis.models), saved to results/checks.json:
- outcome rate by decile of light_change_3m (is the link a straight line?)
- violence-only and controls-only models, to see what each group adds
- gradient boosting with a fixed 300 trees, to check the early-stopping choice
- the lights test in lit localities only (prewar_light above the median)
- how many distinct outbreaks the test rows contain
Run from the repository root:  python -m analysis.checks
"""
import json

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc

from analysis import models as M


def main():
    d = M.load()
    y = "y3"
    train, test = M.split(d, 3)
    both = pd.concat([train, test])
    out = {}

    both["decile"] = pd.qcut(both["light_change_3m"].rank(method="first"), 10, labels=False)
    out["rate_by_light_change_decile"] = (both.groupby("decile")
                                          .agg(median_change=("light_change_3m", "median"), rate=(y, "mean"))
                                          .round(4).reset_index().to_dict("records"))
    out["rate_by_sudden_drop"] = both.groupby("sudden_drop")[y].mean().round(4).to_dict()

    for name, features in [("violence_only", M.VIOLENCE), ("controls_only", M.CONTROLS)]:
        s = M.fit_logit(train, features, y).predict_proba(test[features])[:, 1]
        out[f"logistic_{name}"] = {"pr_auc": round(ap(test[y], s), 4), "roc_auc": round(auc(test[y], s), 3)}

    for version in ["baseline", "baseline_plus_lights"]:
        f = M.VERSIONS[version]
        s = lgb.LGBMClassifier(n_estimators=300, **M.GBM_PARAMS).fit(train[f], train[y]).predict_proba(test[f])[:, 1]
        out[f"gbm_300_trees_{version}"] = {"pr_auc": round(ap(test[y], s), 4), "roc_auc": round(auc(test[y], s), 3)}

    median = d.groupby("locality_id")["prewar_light"].first().median()
    lit_train, lit_test = train[train["prewar_light"] > median], test[test["prewar_light"] > median]
    for version in ["baseline", "baseline_plus_lights"]:
        f = M.VERSIONS[version]
        s = M.fit_logit(lit_train, f, y).predict_proba(lit_test[f])[:, 1]
        out[f"lit_localities_logistic_{version}"] = {"pr_auc": round(ap(lit_test[y], s), 4),
                                                     "roc_auc": round(auc(lit_test[y], s), 3)}

    out["distinct_outbreaks_in_test_years"] = int(d[(d["year"] >= M.TEST_START)]["new_outbreak"].sum())
    with open("results/checks.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
