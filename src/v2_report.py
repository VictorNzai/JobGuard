"""Final numbers + figures for the deployed v2 model (EMSCAD + 300 fake / 900 legit synthetic ads).
python src/v2_report.py -> reports/v2_results.json, reports/figures/13_v2_pr_curve.png, 14_v2_confusion_roc.png"""
import json, sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import build_features  # noqa: E402

raw = pd.read_csv(ROOT / "data" / "fake_job_postings.csv")
df = build_features(raw.drop_duplicates(subset=[c for c in raw.columns if c != "job_id"]))
y = df["fraudulent"]; X = df.drop(columns=["fraudulent"])
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.15 / 0.85, stratify=y_tmp, random_state=42)

art = joblib.load(ROOT / "models" / "jobguard_v2.joblib"); m, thr = art["model"], art["threshold"]
pv, pt = m.predict_proba(X_val)[:, 1], m.predict_proba(X_test)[:, 1]
pred = pt >= thr
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
out = dict(threshold=thr, test_n=len(y_test), test_fake=int(y_test.sum()), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
           precision=tp / (tp + fp), recall=tp / (tp + fn), val_pr_auc=average_precision_score(y_val, pv),
           test_pr_auc=average_precision_score(y_test, pt), test_roc_auc=roc_auc_score(y_test, pt))
out["f1"] = 2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"])
res = X_test.assign(y=y_test.values, pred=pred.astype(int)); res["region"] = np.where(res.country == "US", "US", "Non-US")
out["bias"] = {}
for g, d in res.groupby("region"):
    a, b, c, e = confusion_matrix(d.y, d.pred, labels=[0, 1]).ravel()
    out["bias"][g] = dict(n=len(d), recall=e / max(e + c, 1), fpr=b / max(b + a, 1))
out = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in out.items()}
(ROOT / "reports" / "v2_results.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

pr, rc, th = precision_recall_curve(y_val, pv); i = int(np.argmin(np.abs(th - thr)))
fig, ax = plt.subplots(figsize=(5.5, 4)); ax.plot(rc, pr, color="#0f766e"); ax.scatter(rc[i], pr[i], color="#dc2626", zorder=3, label=f"threshold {thr:.3f}")
ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (validation, v2)"); ax.legend(); ax.grid(alpha=.3)
fig.savefig(ROOT / "reports" / "figures" / "13_v2_pr_curve.png", dpi=150, bbox_inches="tight")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Real", "Fake"], ax=ax[0], colorbar=False, cmap="Greens")
RocCurveDisplay.from_predictions(y_test, pt, ax=ax[1], plot_chance_level=True); ax[0].set_title("Confusion matrix (test, v2)"); ax[1].set_title("ROC (test, v2)")
fig.savefig(ROOT / "reports" / "figures" / "14_v2_confusion_roc.png", dpi=150, bbox_inches="tight")
print(json.dumps(out, indent=1, default=float))
