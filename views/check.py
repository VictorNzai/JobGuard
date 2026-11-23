import csv
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from demos import BLANK, DEMOS  # noqa: E402
from explain import highlight, top_words  # noqa: E402
from features import build_features  # noqa: E402
from fetch import fetch_job  # noqa: E402
from redflags import SEVERE, load_stats, triggered  # noqa: E402

st.markdown("<style>.block-container {max-width: 820px;}</style>", unsafe_allow_html=True)
st.title("Check a job ad")
st.caption("Paste the ad or a link to it. You get a fraud score, the words that raised it, and the red flags behind it.")


@st.cache_resource
def load():
    return joblib.load(ROOT / "models" / "jobguard_v2.joblib")


STATS = load_stats()
art = load()
model, threshold = art["model"], art["threshold"]
SUSPICIOUS_FROM = 0.5
FEEDBACK = ROOT / "data" / "feedback.csv"  # ponytail: local file, wiped on redeploy on free hosts; swap for a sheet/DB if you keep it
LEVELS = {  # level -> (st function, icon, headline)
    "bad": (st.error, ":material/gpp_bad:", "Likely fake. Do not apply or pay anything."),
    "warn": (st.warning, ":material/warning:", "Suspicious. Verify the employer before you go further."),
    "good": (st.success, ":material/verified_user:", "Looks legitimate. That is not a guarantee, so still check the company."),
}
URL_RE = re.compile(r"https?://\S+")
ss = st.session_state
for k, v in BLANK.items():
    ss.setdefault(k, v)


def fill(d):  # on_click callback: runs before the widgets are drawn
    ss.update(d)


def run_check():
    for k in ("result", "err", "notice", "sent"):
        ss.pop(k, None)
    text = ss.ad.strip()
    source = ""
    if URL_RE.fullmatch(text):  # the box holds only a link: read the page, then check what it says
        try:
            with st.spinner("Reading the page…"):
                page = fetch_job(text)
        except Exception as e:  # network, SSL, bad URL, blocked page
            ss.err = f"Couldn't read that link ({e}). Paste the ad text instead."
            return
        source = text
        ss.update(dict(BLANK, title=page["title"], location=page["location"], ad=page["description"],
                       salary=page["salary"], employment_type=page["employment_type"], logo=page["has_image"]))
        ss.notice = ("Page read and the ad text filled in below. The logo box is ticked only if the page has an image, "
                     "so check the details tab.")
    if not (ss.title.strip() or ss.ad.strip()):
        ss.err = "Paste the ad text or a link first."
        return
    row = pd.DataFrame([{
        "title": ss.title, "location": ss.location, "company_profile": ss.company_profile or None,
        "description": f"{ss.ad} {ss.link}".strip(), "requirements": ss.requirements or None,
        "benefits": ss.benefits or None, "salary_range": ss.salary or None,
        "employment_type": ss.employment_type or None, "telecommuting": int(ss.remote),
        "has_company_logo": int(ss.logo), "has_questions": int(ss.questions),
        "required_experience": None, "required_education": None, "industry": None, "function": None,
    }])
    feats = build_features(row)
    p = float(model.predict_proba(feats)[0, 1])
    data_flags, known = triggered(row, STATS)
    known.sort(key=lambda k: k[0] not in SEVERE)  # worst first
    if p >= threshold:
        level = "bad"
    elif p >= SUSPICIOUS_FROM or any(i in SEVERE for i, _ in known) or len(known) >= 2:
        level = "warn"  # the model is blind to M-Pesa/fee scams, so the rule tier can raise the verdict
    else:
        level = "good"
    shown = "\n\n".join(t for t in (ss.title, ss.company_profile, ss.ad, ss.requirements, ss.benefits) if t.strip())
    ss.result = dict(p=p, level=level, data_flags=data_flags, known=known, words=top_words(model, feats),
                     text=shown, source=source)


if ss.pop("autorun", False):  # set by the Home page demo button
    run_check()

slot = st.container()  # results render here, above the input, so they are on screen after a check

with st.expander("The ad to check", expanded="result" not in ss, icon=":material/edit_note:"):
    tab_ad, tab_more = st.tabs(["Ad text or link", "Extra details (optional)"])
    with tab_ad:
        st.text_area("Job ad", key="ad", height=200, label_visibility="collapsed",
                     placeholder="Paste the job ad here, or just a link to it (public pages only; "
                                 "Facebook, LinkedIn and WhatsApp links need a login, so paste their text instead).")
        b1, b2, b3 = st.columns(3)
        b1.button("Demo: fake ad", on_click=fill, args=(DEMOS["fake"],), icon=":material/bug_report:", width="stretch")
        b2.button("Demo: real ad", on_click=fill, args=(DEMOS["real"],), icon=":material/work:", width="stretch")
        b3.button("Clear", on_click=fill, args=(BLANK,), icon=":material/delete:", width="stretch")
    with tab_more:
        st.caption("Filling these in makes the score more accurate. All optional.")
        st.text_input("Job title", key="title")
        c1, c2 = st.columns(2)
        c1.text_input("Location (e.g. KE, Nairobi)", key="location")
        c2.text_input("Apply link or website in the ad", key="link")
        st.text_area("Company profile", height=80, key="company_profile")
        st.text_area("Requirements", height=80, key="requirements")
        st.text_area("Benefits", height=60, key="benefits")
        s1, s2 = st.columns(2)
        s1.text_input("Salary range", key="salary")
        s2.selectbox("Employment type", ["", "Full-time", "Part-time", "Contract", "Temporary", "Other"], key="employment_type")
        k1, k2, k3 = st.columns(3)
        k1.checkbox("Company logo shown", key="logo")
        k2.checkbox("Has screening questions", key="questions")
        k3.checkbox("Remote work", key="remote")
    st.button("Check this ad", type="primary", on_click=run_check, icon=":material/search:", width="stretch")


def md(s):  # `$` pairs would turn into maths in markdown
    return s.replace("$", "\\$")


def summary(r):
    lines = [f"JobGuard: {LEVELS[r['level']][2]}", f"Fraud score: {r['p']:.0%}"]
    if ss.title.strip():
        lines.append(f"Ad: {ss.title.strip()}")
    if r["words"]:
        lines.append("Words that raised the score: " + ", ".join(r["words"]))
    lines += [f"- {d['label']}" for d in r["data_flags"]] + [f"- {label}" for _, label in r["known"]]
    lines.append("A warning, not proof. Trained on 2012-2014 mostly-US ads.")
    return "\n".join(lines)


def save_feedback(r, truth):
    FEEDBACK.parent.mkdir(exist_ok=True)
    new = not FEEDBACK.exists()
    with FEEDBACK.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time_utc", "score", "verdict", "user_says", "text"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), f"{r['p']:.3f}", r["level"], truth, r["text"]])
    ss.sent = True


with slot:
    if "err" in ss:
        st.error(ss.err, icon=":material/error:")
    if "notice" in ss:
        st.info(ss.notice, icon=":material/check_circle:")
    r = ss.get("result")
    if r:
        show, icon, headline = LEVELS[r["level"]]
        with st.container(border=True):
            c1, c2 = st.columns([1, 3], vertical_alignment="center")
            c1.markdown(f'<div class="pct {r["level"]}">{r["p"]:.0%}</div><div class="sub">fraud score</div>',
                        unsafe_allow_html=True)
            with c2:
                show(headline, icon=icon)
                st.progress(min(r["p"], 1.0))
                st.caption(f"Fake alert from {threshold:.0%}. Suspicious from {SUSPICIOUS_FROM:.0%}, "
                           "or when the scam tricks below show up.")

        if r["words"] and r["level"] != "good":  # on a clean ad these words would just be noise
            st.markdown("##### Words that raised the score")
            st.markdown(" ".join(f":red-badge[{w}]" for w in r["words"]))
            with st.expander("Show them in the ad", icon=":material/format_ink_highlighter:"):
                st.markdown(f'<div class="adtext">{highlight(r["text"], r["words"])}</div>', unsafe_allow_html=True)

        if r["data_flags"]:
            st.markdown("##### Red flags backed by our dataset")
            for d in r["data_flags"]:
                with st.container(border=True):
                    st.markdown(f":material/flag: **{md(d['label'])}**")
                    st.caption(f"In the dataset about **{d['rate_distinct']:.0%}** of {d['n_distinct']:,} ads like this were fake "
                               f"(typical ad: {STATS['baseline_distinct']:.1%}).")
        if r["known"]:
            st.markdown("##### Common scam tricks")
            for fid, label in r["known"]:
                with st.container(border=True):
                    st.markdown(f":material/gpp_maybe: **{md(label)}**" + ("  :red-badge[serious]" if fid in SEVERE else ""))
            st.caption("From consumer-protection guidance on job scams, not our dataset, which can't measure them.")
        if not (r["data_flags"] or r["known"]):
            st.info("No red flags found." + (" The model still scored it high from its overall wording, so check the "
                                             "company independently." if r["level"] == "bad" else ""))

        if r["level"] != "good":
            with st.expander("What to do next", expanded=True, icon=":material/checklist:"):
                st.markdown(
                    "1. **Don't pay anything.** Real employers don't charge registration, training, medical, visa or "
                    "\"refundable deposit\" fees.\n"
                    "2. **Hold back your ID, KRA PIN and bank details** until you've confirmed the employer through "
                    "the company's own website or phone number, not the contacts in the ad.\n"
                    "3. **Search the company name with the word \"scam\"** and check the email domain matches its website. "
                    "Gmail or WhatsApp-only contact is a warning sign.\n"
                    "4. **Overseas job?** Check the agency is licensed with the National Employment Authority (NEA) first.\n"
                    "5. **Already paid or shared details?** Tell your bank or mobile-money provider straight away "
                    "and report it to the police (DCI cybercrime unit).")

        with st.expander("Share or save this result", icon=":material/share:"):
            text = summary(r)
            st.code(text, language=None)
            st.download_button("Download as .txt", text, file_name="jobguard-result.txt", icon=":material/download:")
        with st.expander("Is this score wrong?", icon=":material/rate_review:"):
            if ss.get("sent"):
                st.success("Thanks, this helps improve the model.")
            else:
                truth = st.radio("This ad is actually", ["Legitimate", "Fake"], horizontal=True)
                st.caption("Sends the ad text and your answer to the project so it can be used to retrain the model.")
                st.button("Send feedback", on_click=save_feedback, args=(r, truth), icon=":material/send:")
