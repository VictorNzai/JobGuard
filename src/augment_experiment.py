"""Does adding synthetic modern scams help? Real val/test stay 100% real EMSCAD.
python src/augment_experiment.py   (~5 min) -> reports/augmentation_results.json, models/jobguard_aug.joblib"""
import json, sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import build_features  # noqa: E402
from models import stacking_model  # noqa: E402

COLS = ["title", "location", "company_profile", "description", "requirements", "benefits", "salary_range", "employment_type",
        "telecommuting", "has_company_logo", "has_questions", "required_experience", "required_education", "industry", "function"]
raw = pd.read_csv(ROOT / "data" / "fake_job_postings.csv")
df = build_features(raw.drop_duplicates(subset=[c for c in raw.columns if c != "job_id"]))
y = df["fraudulent"]; X = df.drop(columns=["fraudulent"])
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.15 / 0.85, stratify=y_tmp, random_state=42)

NF, NL = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (None, None)   # optional: how many synthetic fakes / legits
syn = pd.read_csv(ROOT / "data" / "synthetic_train.csv")
if NF:
    syn = pd.concat([syn[syn.fraudulent == 1].head(NF), syn[syn.fraudulent == 0].head(NL)], ignore_index=True)
TAG = f"_{NF}_{NL}" if NF else ""
ho = pd.read_csv(ROOT / "data" / "synthetic_holdout.csv")
S_tr, S_ho = build_features(syn), build_features(ho)
ext = ROOT / "data" / "external"
t_ = pd.read_parquet(ext / "techscams.parquet")
tech = pd.DataFrame({c: None for c in COLS}, index=range(len(t_))); tech["description"] = t_.text_sample.values; tech["title"] = ""
tech[["telecommuting", "has_company_logo", "has_questions"]] = 0
j = pd.read_csv(ext / "jobscams3.csv").dropna(subset=["Fake"])
js = pd.DataFrame({c: None for c in COLS}, index=range(len(j)))
for a, b in [("title", "Title"), ("location", "Location"), ("company_profile", "Company Profile"), ("description", "Description"),
             ("requirements", "Requirements"), ("benefits", "Benefits"), ("salary_range", "Salary Range")]:
    js[a] = j[b].values
js[["telecommuting", "has_company_logo", "has_questions"]] = 0
TECH, JS = build_features(tech), build_features(js)


def evaluate(model, thr=None):
    pv = model.predict_proba(X_val)[:, 1]
    if thr is None:
        pr, rc, th = precision_recall_curve(y_val, pv); thr = float(th[np.argmax((2 * pr * rc / (pr + rc + 1e-9))[:-1])])
    pt = model.predict_proba(X_test)[:, 1]; f = pt >= thr
    tp = (f & (y_test == 1)).sum()
    res = dict(threshold=thr, val_pr_auc=average_precision_score(y_val, pv), test_pr_auc=average_precision_score(y_test, pt), test_roc_auc=roc_auc_score(y_test, pt),
               test_precision=tp / max(f.sum(), 1), test_recall=tp / (y_test == 1).sum())
    res["test_f1"] = 2 * res["test_precision"] * res["test_recall"] / (res["test_precision"] + res["test_recall"])
    pt_ = model.predict_proba(TECH)[:, 1]
    res.update(tech_flagged_at_thr=(pt_ >= thr).mean(), tech_flagged_at_0_5=(pt_ >= .5).mean())
    for k, g in t_.assign(p=pt_).groupby("scam_type"):
        res[f"tech_{k.split()[0].lower()}_flagged_at_thr"] = (g.p >= thr).mean()
    pj = model.predict_proba(JS)[:, 1]
    res.update(js_pr_auc=average_precision_score(j.Fake, pj), js_roc_auc=roc_auc_score(j.Fake, pj), js_recall_at_thr=(pj[j.Fake.values == 1] >= thr).mean(),
               js_fpr_at_thr=(pj[j.Fake.values == 0] >= thr).mean())
    ph = model.predict_proba(S_ho)[:, 1]
    res.update(synth_holdout_pr_auc=average_precision_score(S_ho.fraudulent, ph), synth_holdout_recall_at_thr=(ph[S_ho.fraudulent.values == 1] >= thr).mean(),
               synth_holdout_fpr_at_thr=(ph[S_ho.fraudulent.values == 0] >= thr).mean())
    return {k: float(v) for k, v in res.items()}


out = {}
base = joblib.load(ROOT / "models" / "jobguard.joblib")
out["baseline"] = evaluate(base["model"], base["threshold"])
print("baseline", json.dumps({k: round(v, 3) for k, v in out["baseline"].items()}), flush=True)
assert abs(out["baseline"]["test_pr_auc"] - 0.936) < 0.01, "split doesn't reproduce the notebook"

t0 = time.time()
aug = stacking_model().fit(pd.concat([X_train, S_tr.drop(columns=["fraudulent"])], ignore_index=True), pd.concat([y_train, S_tr.fraudulent], ignore_index=True))
print("trained augmented in", round(time.time() - t0), "s", flush=True)
out["augmented"] = evaluate(aug)
print("augmented", json.dumps({k: round(v, 3) for k, v in out["augmented"].items()}), flush=True)
(ROOT / "reports" / f"augmentation_results{TAG}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
joblib.dump({"model": aug, "threshold": out["augmented"]["threshold"]}, ROOT / "models" / f"jobguard_aug{TAG}.joblib", compress=3)
