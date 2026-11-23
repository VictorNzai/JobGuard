import json
from pathlib import Path

import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from demos import DEMOS  # noqa: E402

R = Path(__file__).parent.parent / "reports"
S = json.loads((R / "dataset_summary.json").read_text(encoding="utf-8"))

st.markdown("""<div class="hero big"><h1>JobGuard</h1>
<p>Fake job ads trick job-seekers into paying "registration fees" or sending ID copies.
JobGuard reads an ad, scores how likely it is to be fake, and shows the red flags behind the score.</p></div>""",
            unsafe_allow_html=True)

if st.button("Try it now: check a fake ad", type="primary", icon=":material/bug_report:", width="stretch"):
    st.session_state.update(DEMOS["fake"], autorun=True)  # the Check page runs it on arrival
    st.switch_page("views/check.py")

a, b, c = st.columns(3)
a.page_link("views/check.py", label="Check your own ad", icon=":material/search:", width="stretch")
b.page_link("views/data.py", label="Explore the data", icon=":material/bar_chart:", width="stretch")
c.page_link("views/findings.py", label="See what we found", icon=":material/lightbulb:", width="stretch")

st.markdown("&nbsp;")
t = st.columns(4)
for col, n, label, cls in [
    (t[0], f"{S['rows']:,}", "real-world job ads in the EMSCAD dataset", ""),
    (t[1], f"{S['fake'] / S['rows']:.1%}", f"are fraudulent ({S['fake']} ads): a heavily imbalanced problem", "red"),
    (t[2], "0.94", "PR-AUC of the final stacked model (validation)", ""),
    (t[3], "90%", "of ads it flags as fake really are (test precision)", ""),
]:
    col.markdown(f'<div class="tile {cls}"><div class="n">{n}</div><div class="t">{label}</div></div>', unsafe_allow_html=True)

st.markdown("### How it works")
s = st.columns(3)
for col, h, txt in [
    (s[0], "1 · Read the ad", "Paste a job link or the ad text. We turn the text into word weights and the form into signals: logo, company profile, salary, contact details."),
    (s[1], "2 · Four models vote, a fifth decides", "Two models read the words, two read the structure. A final model learns how much to trust each."),
    (s[2], "3 · Explain the score", "Red flags like \"pay + no experience needed\" are shown with how often ads like that were fake in our data."),
]:
    with col.container(border=True):
        st.markdown(f"**{h}**")
        st.write(txt)

st.markdown("### Honest limits")
st.info("The training data is from 2012–2014 and mostly American. Scams common in Kenya today (M-Pesa fees, WhatsApp-only contact) "
        "are not in it, so those are flagged by rules, not learned. Use the score as a warning, not proof.")
