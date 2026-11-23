"""Named red flags: one list used by the app AND by the script that measures each flag on EMSCAD.

Two tiers:
  data  - measured on the dataset; the app shows "X% of N ads like this were fake".
  known - common scam patterns from consumer-protection guidance. EMSCAD can't back them
          (emails/phones were anonymised, it predates M-Pesa/WhatsApp scams), so no rate is shown.

Run `python src/redflags.py` (needs data/fake_job_postings.csv) to rewrite reports/red_flag_stats.json.
"""
import json
import re
from pathlib import Path

import pandas as pd

TEXT = ["title", "company_profile", "description", "requirements", "benefits"]
MONEY = r"(?:\$|usd|kes|ksh|kshs|£|€)\s?\d[\d,\.]*|\d[\d,\.]*\s?(?:usd|kes|ksh|kshs|dollars|shillings)"
PER = r"(?:per|a|/|each|every)\s?(?:week|day|hour|month)|weekly|daily|hourly"
NOEXP = r"no (?:prior |previous )?experience|without experience|no qualifications?|no skills? (?:needed|required)|anyone can"
WFH = r"work(?:ing)? from home|home[- ]?based|work at home|work online|online job"
PHONE = r"#phone_|\+\d[\d\s\-]{8,}\d|(?<!\d)0[17]\d{2}[\s\-]?\d{3}[\s\-]?\d{3}(?!\d)"  # raw is lower-cased


def _raw(df):
    return df[TEXT].fillna("").astype(str).agg(" ".join, axis=1)


def _has(df, col):
    return df[col].fillna("").astype(str).str.strip().str.len() > 0


# (id, label, tier, test(df, raw) -> boolean Series). `raw` is the lower-case joined ad text.
FLAGS = [
    ("no_profile", "No company profile", "data", lambda d, r: ~_has(d, "company_profile")),
    ("no_logo", "No company logo", "data", lambda d, r: d["has_company_logo"].fillna(0).astype(int) == 0),
    ("bare", "No company profile, no logo and no requirements", "data",
     lambda d, r: ~_has(d, "company_profile") & ~_has(d, "requirements") & (d["has_company_logo"].fillna(0).astype(int) == 0)),
    ("earn", 'Promises you will "earn" money', "data", lambda d, r: r.str.contains(r"\bearn", regex=True)),
    ("wfh", "Work-from-home / online job", "data", lambda d, r: r.str.contains(WFH, regex=True)),
    ("data_entry", "Data entry / typing job", "data", lambda d, r: r.str.contains(r"data entry|typist|typing", regex=True)),
    ("no_exp", '"No experience needed"', "data", lambda d, r: r.str.contains(NOEXP, regex=True)),
    ("pay_noexp", "Specific pay amount for little or no skill (pay + no experience / work from home / data entry)", "data",
     lambda d, r: r.str.contains(MONEY, regex=True) & r.str.contains(f"{NOEXP}|{WFH}|data entry|typist|typing", regex=True)),
    ("guaranteed", 'Uses "guaranteed"', "data", lambda d, r: r.str.contains(r"guarantee", regex=True)),
    ("easy_income", "Promises easy or unlimited income", "known",
     lambda d, r: r.str.contains(r"passive income|extra income|unlimited (?:income|earning)|financial freedom|easy money", regex=True)),
    ("email", "Asks you to contact an email address", "data",
     lambda d, r: r.str.contains(r"#email_|[\w.+-]+@[\w-]+\.[\w.]+", regex=True)),
    ("phone", "Gives a phone number in the ad", "data", lambda d, r: r.str.contains(PHONE, regex=True)),
    ("caps", "Lots of ALL-CAPS words", "data",
     lambda d, r: d["description"].fillna("").astype(str).str.count(r"\b[A-Z]{4,}\b") >= 5),
    ("title_symbol", "Title contains ! or $", "data", lambda d, r: d["title"].fillna("").astype(str).str.contains(r"[!$]", regex=True)),
    # ---- known patterns: no rate, the dataset cannot measure these ----
    ("fee", "Asks you to pay a fee (registration, processing, training, medical, visa, kit)", "known",
     lambda d, r: r.str.contains(r"(?:registration|processing|application|admin|interview|training|medical|visa|uniform|kit|security) (?:fee|deposit|charge|payment)"
                                 r"|(?:pay|send|deposit|transfer)\W+(?:\w+\W+){0,4}(?:fee|deposit|ksh|kes)|refundable (?:fee|deposit)", regex=True)),
    ("mobile_money", "Mentions M-Pesa, Paybill/Till, Western Union, crypto or gift cards", "known",
     lambda d, r: r.str.contains(r"m-?pesa|pay ?bill|till number|western union|moneygram|bitcoin|crypto|gift card", regex=True)),
    ("chat_contact", "Contact only via WhatsApp / Telegram", "known", lambda d, r: r.str.contains(r"whatsapp|telegram|signal app", regex=True)),
    ("free_email", "Uses a personal email (Gmail, Yahoo, Hotmail) instead of a company address", "known",
     lambda d, r: r.str.contains(r"@(?:gmail|yahoo|hotmail|outlook|ymail)\.", regex=True)),
    ("id_request", "Asks for ID, passport, KRA PIN or bank details up front", "known",
     lambda d, r: r.str.contains(r"(?:send|submit|provide|attach)\W+(?:\w+\W+){0,5}(?:id|passport|kra pin|pin number|bank (?:details|account)|national id)"
                                 r"|id copy|copy of (?:your )?(?:id|passport)", regex=True)),
    ("overseas", "Overseas placement with visa, ticket or guaranteed job", "known",
     lambda d, r: r.str.contains(r"dubai|qatar|saudi|kuwait|oman|abroad|overseas|canada|australia", regex=True)
     & r.str.contains(r"visa|air ?ticket|guaranteed|placement|sponsor", regex=True)),
    ("short_link", "Hides the real website behind a shortened link (bit.ly, tinyurl, etc.)", "known",
     lambda d, r: r.str.contains(r"\b(?:bit\.ly|tinyurl\.com|cutt\.ly|goo\.gl|rb\.gy|is\.gd|t\.ly)/", regex=True)),
    ("urgent", "Pressure to act immediately", "known",
     lambda d, r: r.str.contains(r"urgent|apply (?:now|today|immediately)|hurry|limited (?:slots|vacancies|positions)|only \d+ (?:slots|positions)|first come", regex=True)),
]
SEVERE = {"fee", "mobile_money", "id_request"}  # a single hit is enough to call an ad suspicious
META = {f[0]: (f[1], f[2]) for f in FLAGS}
STATS_PATH = Path(__file__).resolve().parent.parent / "reports" / "red_flag_stats.json"


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean matrix (rows = ads, columns = flag ids). `df` has the raw dataset columns."""
    raw = _raw(df).str.lower()
    # caps test needs original case, so tests receive the original frame too
    return pd.DataFrame({fid: fn(df, raw).astype(bool) for fid, _, _, fn in FLAGS}, index=df.index)


def triggered(row: pd.DataFrame, stats: dict):
    """-> (data_flags, known_flags). Data flags: dicts with measured numbers, sorted by lift. Known flags: (id, label)."""
    hit = evaluate(row).iloc[0]
    data, known = [], []
    for fid, label, tier, _ in FLAGS:
        if not hit[fid]:
            continue
        if tier == "known":
            known.append((fid, label))
        elif fid in stats["flags"]:
            data.append(dict(stats["flags"][fid], id=fid, label=label))
    if any(d["id"] == "bare" for d in data):  # the combined flag already covers these two
        data = [d for d in data if d["id"] not in ("no_profile", "no_logo")]
    data.sort(key=lambda d: -d["lift_distinct"])
    return data, known


def load_stats() -> dict:
    return json.loads(STATS_PATH.read_text(encoding="utf-8"))


def build_stats(csv: str) -> dict:
    df = pd.read_csv(csv)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "job_id"]).reset_index(drop=True)
    # reposted scam templates would inflate every rate, so also count each near-identical description once
    key = df["description"].fillna("").str.lower().str.replace(r"\W+", " ", regex=True).str[:150]
    dd = df[~key.duplicated()].reset_index(drop=True)
    out = {"baseline": float(df.fraudulent.mean()), "baseline_distinct": float(dd.fraudulent.mean()),
           "n": len(df), "n_distinct": len(dd), "flags": {}}
    m_all, m_dd = evaluate(df), evaluate(dd)
    for fid, _, tier, _ in FLAGS:
        if tier != "data":
            continue
        a, b = m_all[fid], m_dd[fid]
        out["flags"][fid] = dict(
            n=int(a.sum()), rate=float(df.fraudulent[a].mean()), lift=float(df.fraudulent[a].mean() / out["baseline"]),
            share_of_fraud=float((a & (df.fraudulent == 1)).sum() / df.fraudulent.sum()),
            n_distinct=int(b.sum()), rate_distinct=float(dd.fraudulent[b].mean()) if b.sum() else 0.0,
            lift_distinct=float(dd.fraudulent[b].mean() / out["baseline_distinct"]) if b.sum() else 0.0,
            fraud_templates=int((b & (dd.fraudulent == 1)).sum()))
    return out


if __name__ == "__main__":
    s = build_stats(str(Path(__file__).resolve().parent.parent / "data" / "fake_job_postings.csv"))
    print(f"baseline {s['baseline']:.3f} ({s['n']} ads) | one-per-template baseline {s['baseline_distinct']:.3f} ({s['n_distinct']} ads)")
    print(f"{'flag':14s} {'n':>5s} {'fake%':>6s} {'lift':>5s} | {'n_dist':>6s} {'fake%':>6s} {'lift':>5s} {'fakes':>5s} | share_of_fraud")
    for k, v in sorted(s["flags"].items(), key=lambda kv: -kv[1]["lift_distinct"]):
        print(f"{k:14s} {v['n']:5d} {v['rate']:6.1%} {v['lift']:5.1f} | {v['n_distinct']:6d} {v['rate_distinct']:6.1%} "
              f"{v['lift_distinct']:5.1f} {v['fraud_templates']:5d} | {v['share_of_fraud']:.0%}")
    STATS_PATH.write_text(json.dumps(s, indent=1), encoding="utf-8")
    # self-check: the Kenyan-style scam must trip the known tier, a clean ad must trip nothing
    bad = pd.DataFrame([dict(title="Office Assistant", company_profile=None, description="Pay KES 1,500 registration fee via M-Pesa Paybill 123456. "
                         "WhatsApp 0712 345 678 or email hr.jobs@gmail.com. Send copy of your ID. URGENT apply now!", requirements=None,
                         benefits=None, has_company_logo=0)])
    got = evaluate(bad).iloc[0]
    assert all(got[k] for k in ("fee", "mobile_money", "chat_contact", "free_email", "id_request", "urgent", "phone", "email")), got[got].index.tolist()
    assert evaluate(pd.DataFrame([dict(title="Job", company_profile=None, description="Apply at http://bit.ly/abc", requirements=None, benefits=None, has_company_logo=0)])).iloc[0]["short_link"]
    ok = pd.DataFrame([dict(title="Nurse", company_profile="Mater Hospital is a leading hospital serving Nairobi since 1962.",
                            description="Provide nursing care on the ward and keep patient records.", requirements="Diploma in nursing, KRCHN registration.",
                            benefits="Medical cover", has_company_logo=1)])
    assert not evaluate(ok).iloc[0].any(), evaluate(ok).iloc[0][lambda s: s].index.tolist()
    print("self-check ok")
