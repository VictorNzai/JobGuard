import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features import build_features  # noqa: E402
from fetch import fetch_job  # noqa: E402
from redflags import load_stats, triggered  # noqa: E402

st.markdown("<style>.block-container {max-width: 820px;}</style>", unsafe_allow_html=True)
st.title("Check a job ad")
st.caption("Paste a link or the ad text. You get a fraud score and the red flags behind it.")

@st.cache_resource
def load():
    return joblib.load(Path(__file__).parent.parent / "models" / "jobguard_v2.joblib")


STATS = load_stats()
art = load()
model, threshold = art["model"], art["threshold"]

BLANK = dict(title="", location="", link="", company_profile="", description="", requirements="", benefits="",
             salary="", employment_type="", logo=False, questions=False, remote=False)
DEMOS = {
    "fake": dict(BLANK, title="Data Entry Clerk - Work From Home", location="KE, Nairobi",
                 link="http://bit.ly/earn-fast-apply", remote=True,
                 description="Earn $500 weekly from home! No experience needed. Send your CV and ID copy to hr.jobs@gmail.com today!"),
    "real": dict(BLANK, title="Senior Software Engineer", location="US, CA, San Francisco", employment_type="Full-time",
                 link="https://www.examplecorp.com/careers/senior-software-engineer", logo=True, questions=True,
                 company_profile="We are a growing software company with 200 employees building payments tools for clients worldwide.",
                 description="Design and build backend services in Python. Work with product and design teams to ship features. Participate in code reviews and mentor junior engineers.",
                 requirements="5+ years of experience with Python, SQL and cloud platforms. Bachelor's degree in computer science or equivalent.",
                 benefits="Health insurance, 401k, flexible hours."),
}


def fill(d):  # on_click callback: runs before the widgets are drawn
    st.session_state.update(d)


for k, v in BLANK.items():
    st.session_state.setdefault(k, v)

# ---- option 1: check a link ----
with st.container(border=True):
    st.markdown('<span class="step">Option 1</span>  **Check a job link**', unsafe_allow_html=True)
    u1, u2 = st.columns([4, 1], vertical_alignment="bottom")
    url = u1.text_input("Job posting URL", key="url", placeholder="https://… link to the job ad",
                        label_visibility="collapsed")
    fetch_clicked = u2.button("Fetch & check", type="primary", width="stretch")
    st.caption("Works for public pages such as job boards and company career pages. Pages that need a login "
               "(Facebook, LinkedIn, WhatsApp) can't be read, so paste the text below instead.")

auto = False
if fetch_clicked:
    if not url.strip():
        st.warning("Paste a link first.")
    else:
        try:
            with st.spinner("Reading the page…"):
                page = fetch_job(url.strip())
            st.session_state.update(dict(
                BLANK, title=page["title"], location=page["location"], description=page["description"],
                salary=page["salary"], employment_type=page["employment_type"], logo=page["has_image"]))
            st.success("Page read. The fields below were filled from it (the logo box is ticked only if the page "
                       "has an image). Check them, then see the result underneath.")
            auto = True
        except Exception as e:  # network, SSL, bad URL, blocked page
            st.error(f"Couldn't fetch that link: {e}")

# ---- option 2: paste the ad ----
with st.container(border=True):
    st.markdown('<span class="step">Option 2</span>  **Paste the ad text**', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    b1.button("Load demo: fake ad", on_click=fill, args=(DEMOS["fake"],), width="stretch")
    b2.button("Load demo: real ad", on_click=fill, args=(DEMOS["real"],), width="stretch")
    b3.button("Clear", on_click=fill, args=(BLANK,), width="stretch")

    with st.form("ad", border=False):
        title = st.text_input("Job title", key="title")
        cl1, cl2 = st.columns(2)
        location = cl1.text_input("Location (e.g. KE, Nairobi)", key="location")
        link = cl2.text_input("Apply link or website in the ad (optional)", key="link")
        company_profile = st.text_area("Company profile", height=80, key="company_profile")
        description = st.text_area("Description", height=150, key="description")
        requirements = st.text_area("Requirements", height=80, key="requirements")
        benefits = st.text_area("Benefits", height=60, key="benefits")
        s1, s2 = st.columns(2)
        salary = s1.text_input("Salary range", key="salary")
        employment_type = s2.selectbox("Employment type", ["", "Full-time", "Part-time", "Contract", "Temporary", "Other"],
                                       key="employment_type")
        c1, c2, c3 = st.columns(3)
        logo = c1.checkbox("Company logo shown", key="logo")
        questions = c2.checkbox("Has screening questions", key="questions")
        remote = c3.checkbox("Remote work", key="remote")
        go = st.form_submit_button("Check this ad", type="primary", width="stretch")

if go or auto:
    if not (title.strip() or description.strip()):
        st.warning("Add at least a title or a description.")
        st.stop()
    row = pd.DataFrame([{
        "title": title, "location": location, "company_profile": company_profile or None,
        "description": f"{description} {link}".strip(), "requirements": requirements or None, "benefits": benefits or None,
        "salary_range": salary or None, "employment_type": employment_type or None,
        "telecommuting": int(remote), "has_company_logo": int(logo), "has_questions": int(questions),
        "required_experience": None, "required_education": None, "industry": None, "function": None,
    }])
    feats = build_features(row)
    p = float(model.predict_proba(feats)[0, 1])
    fake = p >= threshold
    color = "#dc2626" if fake else "#059669"
    st.markdown(f"""
<div class="verdict {'bad' if fake else 'good'}">
  <div class="pct">{p:.0%}</div>
  <div style="flex:1"><div class="lbl">{'⚠️ Likely FAKE' if fake else '✅ Looks legitimate'}</div>
  <div class="sub">Fraud score · alert threshold {threshold:.0%}</div>
  <div class="bar"><div style="width:{min(p, 1) * 100:.0f}%; background:{color}"></div></div></div>
</div>""", unsafe_allow_html=True)

    data_flags, known_flags = triggered(row, STATS)
    st.markdown("**Why?** The score comes from a model that reads the ad's wording and structure. These red flags are "
                "the patterns we can point to.")
    if data_flags:
        st.markdown("##### Red flags backed by our dataset")
        for d in data_flags:
            st.markdown(
                f'<div class="flag"><b>{d["label"]}</b><br><span class="why">In the dataset, about <b>{d["rate_distinct"]:.0%}</b> of '
                f'{d["n_distinct"]:,} ads like this were fake (typical ad: {STATS["baseline_distinct"]:.1%}).</span></div>',
                unsafe_allow_html=True)
    if known_flags:
        st.markdown("##### Common scam tricks")
        for k in known_flags:
            st.markdown(f'<div class="flag known"><b>{k}</b></div>', unsafe_allow_html=True)
        st.caption("These come from consumer-protection guidance on job scams, not from our dataset, which can't measure them.")
    if not (data_flags or known_flags):
        st.info("No red flags found." + (" The model still scored it high from its overall wording, so check the company "
                                         "independently." if fake else ""))
    elif fake and not data_flags:
        st.caption("The model also reacted to the ad's overall wording, not just the flags above.")
