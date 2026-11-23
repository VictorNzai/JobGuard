import json
from pathlib import Path

import pandas as pd
import streamlit as st

S = json.loads((Path(__file__).parent.parent / "reports" / "dataset_summary.json").read_text(encoding="utf-8"))
REAL, FAKE = "#0f766e", "#dc2626"

st.title("The data")
st.caption("EMSCAD (Employment Scam Aegean Dataset, University of the Aegean): job ads scraped from the web and labelled real or fraudulent.")

t = st.columns(4)
for col, n, label in [(t[0], f"{S['raw_rows']:,}", "ads in the raw file"), (t[1], f"{S['duplicates']}", "duplicates removed before splitting"),
                      (t[2], f"{S['rows']:,}", "ads used"), (t[3], f"{S['fake']}", f"fraudulent ({S['fake'] / S['rows']:.1%})")]:
    col.markdown(f'<div class="tile"><div class="n">{n}</div><div class="t">{label}</div></div>', unsafe_allow_html=True)
st.markdown("")


def rate_chart(rows, title, note, horizontal=True):
    df = pd.DataFrame(rows).assign(rate=lambda d: d.rate * 100)
    df["label"] = df.group + "  (n=" + df.n.map("{:,}".format) + ")"
    st.markdown(f"**{title}**")
    st.bar_chart(df.set_index("label")["rate"], horizontal=horizontal, color=FAKE, y_label="% of ads that were fake", x_label="")
    st.caption(note)


tab1, tab2, tab3 = st.tabs(["Balance & missing fields", "Who gets targeted", "Ad length"])

with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Class balance**")
        st.bar_chart(pd.DataFrame({"ads": [S["real"], S["fake"]]}, index=["Real", "Fake"]), color=REAL, x_label="", y_label="ads")
        st.caption("Only 4.9% are fake, so a model that always says \"real\" is 95% accurate and useless. That is why we judge on PR-AUC and F1, not accuracy.")
    with c2:
        st.markdown("**Fields left blank, real vs fake ads**")
        m = pd.DataFrame({k: {"Real": v["real"] * 100, "Fake": v["fake"] * 100} for k, v in S["missing_by_class"].items()}).T
        m = m.sort_values("Fake", ascending=False)
        st.bar_chart(m, horizontal=True, color=[REAL, FAKE], y_label="% missing", x_label="")
        st.caption("Scam ads skip the company profile far more often, so we turned missingness into features (has_company_profile and so on).")

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        logo = {r["group"]: r for r in S["by_logo"]}
        st.markdown("**Company logo**")
        a, b = st.columns(2)
        a.markdown(f'<div class="tile red"><div class="n">{logo["0"]["rate"]:.0%}</div><div class="t">of ads <b>without</b> a logo were fake</div></div>', unsafe_allow_html=True)
        b.markdown(f'<div class="tile"><div class="n">{logo["1"]["rate"]:.0%}</div><div class="t">of ads <b>with</b> a logo were fake</div></div>', unsafe_allow_html=True)
        st.caption("The single strongest structural signal in the data.")
    with c2:
        rate_chart(S["by_employment"], "Employment type", "Fraud rate by the listed employment type.")
    c1, c2 = st.columns(2)
    with c1:
        rate_chart(S["by_country"], "Top 10 countries by fraud rate", "Only countries with at least 50 ads. Most ads are American.")
    with c2:
        rate_chart(S["by_industry"], "Top 10 industries by fraud rate", "Only industries with at least 50 ads.")

with tab3:
    d = S["desc_len"]
    df = pd.DataFrame({k: {f"p{int(float(q) * 100)}": v for q, v in d[k].items()} for k in d}).rename(columns=str.title)
    df.index.name = "percentile"
    st.markdown("**Description length (characters)**")
    st.dataframe(df.style.format("{:,.0f}"), width="stretch")
    st.caption("Percentiles of description length. Real and fake ads overlap heavily, so length alone isn't a strong signal.")

with st.expander("Synthetic ads added to training"):
    st.markdown("300 fake and 900 legitimate generated ads (modern scam types) were added to the training set only. "
                "Validation and test sets are real EMSCAD. See *How we built it*.")

with st.expander("Cleaning steps we applied"):
    st.markdown("""
1. Dropped 281 duplicate ads **before** splitting, so copies can't leak from train into test.
2. Dropped `job_id` (identifier) and `department` (65% missing).
3. Filled missing text with empty strings and missing categories with "Unknown".
4. Turned missingness into features: `has_salary`, `has_company_profile`, `has_requirements`, `has_benefits`.
5. Extracted `country` from the location.
6. Cleaned the text: replaced the anonymised URL / EMAIL / PHONE tokens with words, removed digits and punctuation, lowercased.
7. Added counts of links, emails, phones, description length, capital-letter ratio and `!` count.
""")
