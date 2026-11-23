import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
STATS = json.loads((ROOT / "reports" / "red_flag_stats.json").read_text(encoding="utf-8"))
V2 = json.loads((ROOT / "reports" / "v2_results.json").read_text(encoding="utf-8"))
AUG = json.loads((ROOT / "reports" / "augmentation_results_300_900.json").read_text(encoding="utf-8"))

st.title("Findings")

st.markdown("### Final test results")
st.caption(f"{V2['test_n']:,} held-out real ads ({V2['test_fake']} fake), final model (v2) at threshold {V2['threshold']:.3f}. Source: src/v2_report.py.")
t = st.columns(4)
for col, n, label in [(t[0], f"{V2['precision']:.0%}", f"precision: of ads flagged fake, {V2['precision']:.0%} really were"), (t[1], f"{V2['recall']:.0%}", f"recall: of all fake ads, {V2['recall']:.0%} were caught"),
                      (t[2], f"{V2['f1']:.2f}", "F1 on the fake class"), (t[3], f"{V2['test_roc_auc']:.3f}", f"ROC-AUC (PR-AUC {V2['test_pr_auc']:.3f})")]:
    col.markdown(f'<div class="tile"><div class="n">{n}</div><div class="t">{label}</div></div>', unsafe_allow_html=True)

st.markdown("### Red flags: how often were ads like this fake?")
names = {"pay_noexp": "Pay amount + no experience / WFH / data entry", "bare": "No profile, no logo, no requirements",
         "no_exp": '"No experience needed"', "earn": '"Earn" money', "data_entry": "Data entry / typing", "no_logo": "No company logo",
         "no_profile": "No company profile", "guaranteed": '"Guaranteed"', "wfh": "Work from home", "phone": "Phone number in ad",
         "title_symbol": "! or $ in title", "caps": "Lots of ALL-CAPS", "email": "Contact email in ad"}
df = pd.DataFrame([dict(flag=names[k], rate=v["rate_distinct"] * 100, n=v["n_distinct"]) for k, v in STATS["flags"].items()])
df = df.sort_values("rate", ascending=False)
df["label"] = df.flag + "  (n=" + df.n.map("{:,}".format) + ")"
st.bar_chart(df.set_index("label")["rate"], horizontal=True, color="#dc2626", y_label="% of matching ads that were fake", x_label="")
st.caption(f"The typical ad is {STATS['baseline_distinct']:.1%} fake. Each reposted scam template is counted once so repeats don't inflate the rates.")

c1, c2 = st.columns(2)
with c1:
    st.markdown("### What surprised us")
    st.markdown("""
- **A high salary is not a red flag in this data.** Median top-of-range: 45k for fake vs 50k for real ads. Fake ads even list a salary more often (26% vs 16%).
- **Missing information beats suspicious wording.** No company profile or logo is the most common signal across fake ads (about two thirds of them).
- **Fraud clusters in templates.** The same scam is reposted dozens of times, which inflates naive scores.
""")
with c2:
    st.markdown("### Bias check: US vs non-US")
    st.dataframe(pd.DataFrame({"Ads": [V2["bias"][k]["n"] for k in ("US", "Non-US")], "Recall": [V2["bias"][k]["recall"] for k in ("US", "Non-US")],
                               "False-positive rate": [V2["bias"][k]["fpr"] for k in ("US", "Non-US")]}, index=["US", "Non-US"]).style.format({"Recall": "{:.1%}", "False-positive rate": "{:.1%}"}),
                 width="stretch")
    st.markdown("The model catches **fewer scams outside the US** (72% vs 80%). This is sample bias: the training data is mostly American ads. "
                "It matters for Kenyan users.")

st.markdown("### Does it handle modern scams? (outside check)")
B, A = AUG["baseline"], AUG["augmented"]
chk = pd.DataFrame({"EMSCAD only (v1)": [B["test_pr_auc"], B["test_f1"], B["js_recall_at_thr"], B["js_pr_auc"], B["tech_phishing_flagged_at_thr"]],
                    "+ synthetic ads (v2, deployed)": [A["test_pr_auc"], A["test_f1"], A["js_recall_at_thr"], A["js_pr_auc"], A["tech_phishing_flagged_at_thr"]]},
                   index=["Real EMSCAD test PR-AUC", "Real EMSCAD test F1", "Fake ads caught in a 47-ad modern set", "PR-AUC on that set", "Phishing-interview messages caught"])
st.dataframe(chk.style.format("{:.2f}"), width="stretch")
st.caption("EMSCAD has too few modern scams, so we added synthetic fake AND legitimate ads (written by the same generator so style is not a giveaway). "
           "Cost: about 1 F1 point on the old data. Gain: more modern scams caught. The modern set is tiny (19 fakes), so treat the gain as encouraging, not proven. "
           "Scores on synthetic-only data are meaningless because the generator wrote both sets.")

st.markdown("### Limitations and next steps")
st.markdown("""
- Old, US-centric data. Collect and label Kenyan ads (job boards for real, WhatsApp forwards for scams) and test or retrain on them.
- Scam tricks common in Kenya (M-Pesa fees, WhatsApp-only contact, overseas placement) are flagged by hand-written rules, not learned.
- Ideas: a WhatsApp / Telegram bot, a browser extension, a grouped-split leakage test, hyperparameter tuning.
""")
