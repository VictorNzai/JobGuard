"""Precompute the numbers the dashboard pages chart, so the app doesn't need data/ (gitignored).
Run: python src/summary.py  -> reports/dataset_summary.json"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_MISSING = ["salary_range", "department", "required_education", "benefits", "required_experience", "function",
               "industry", "employment_type", "company_profile", "requirements", "location"]

raw = pd.read_csv(ROOT / "data" / "fake_job_postings.csv")
df = raw.drop_duplicates(subset=[c for c in raw.columns if c != "job_id"]).reset_index(drop=True)
df["country"] = df.location.fillna("").str.split(",").str[0].str.strip().replace("", "Unknown")


def rate_by(col, min_n=1, top=None):
    g = df.groupby(col, dropna=False).fraudulent.agg(n="size", fake="sum")
    g = g[g.n >= min_n].assign(rate=lambda x: x.fake / x.n).sort_values("rate", ascending=False)
    g = g.head(top) if top else g
    return [dict(group=str(i), n=int(r.n), rate=float(r.rate)) for i, r in g.iterrows()]


out = dict(
    raw_rows=len(raw), duplicates=len(raw) - len(df), rows=len(df), columns=raw.shape[1],
    real=int((df.fraudulent == 0).sum()), fake=int((df.fraudulent == 1).sum()),
    missing_raw={c: float(raw[c].isna().mean()) for c in RAW_MISSING},
    missing_by_class={c: {"real": float(df[df.fraudulent == 0][c].isna().mean()), "fake": float(df[df.fraudulent == 1][c].isna().mean())}
                      for c in RAW_MISSING},
    by_logo=rate_by("has_company_logo"), by_telecommuting=rate_by("telecommuting"), by_questions=rate_by("has_questions"),
    by_employment=rate_by("employment_type"), by_experience=rate_by("required_experience"),
    by_education=rate_by("required_education", min_n=50),
    by_country=rate_by("country", min_n=50, top=10), by_industry=rate_by("industry", min_n=50, top=10),
    desc_len={k: {q: float(df[df.fraudulent == v].description.fillna("").str.len().quantile(q)) for q in (.1, .25, .5, .75, .9)}
              for k, v in (("real", 0), ("fake", 1))},
)
(ROOT / "reports" / "dataset_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print({k: v for k, v in out.items() if not isinstance(v, (list, dict))})
print(out["by_logo"], out["by_country"][:3])
