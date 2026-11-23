from pathlib import Path

import pandas as pd
import streamlit as st

R = Path(__file__).parent.parent / "reports"

st.title("How we built it")
st.caption("Following CRISP-DM: understand the problem and data, prepare it, model, evaluate, deploy.")

st.markdown("### The model: a stacked ensemble")
st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#0f766e", fillcolor="#ecfdf5"];
  ad [label="Job ad", fillcolor="#ffffff"];
  text [label="Text → TF-IDF"]; meta [label="Metadata\n(logo, profile, salary, missing fields)"];
  nb [label="Naive Bayes"]; lr [label="Logistic Regression"]; rf [label="Random Forest"]; gb [label="Gradient Boosting"];
  meta_lr [label="Meta Logistic Regression\n(learns who to trust)", fillcolor="#0f766e", fontcolor="white"];
  out [label="Fraud score\n+ tuned threshold 0.906", fillcolor="#fef2f2", color="#dc2626"];
  ad -> text; ad -> meta; text -> nb; text -> lr; meta -> lr; meta -> rf; text -> rf; meta -> gb;
  nb -> meta_lr; lr -> meta_lr; rf -> meta_lr; gb -> meta_lr; meta_lr -> out;
}""")
st.caption("Each model sees a different view of the ad. The meta-model combines their probabilities, trained with 5-fold cross-validation so it never sees a model's scores on data that model was trained on.")

st.markdown("### Model comparison (validation set, EMSCAD only)")
df = pd.read_csv(R / "model_comparison.csv").set_index("Model")
c1, c2 = st.columns([3, 2])
with c1:
    st.dataframe(df.style.format("{:.3f}", subset=df.columns[:-1]).format("{:.0f}", subset=["Train s"])
                 .highlight_max(subset=["PR-AUC", "ROC-AUC"], color="#a7f3d0"), width="stretch")
with c2:
    st.bar_chart(df["PR-AUC"].sort_values(), horizontal=True, color="#0f766e", y_label="PR-AUC", x_label="")
st.markdown("""
- **PR-AUC is the headline metric**: with 4.9% fraud, it measures how well fakes are ranked above real ads. Stacking is best.
- **Stacking's F1 at 0.5 looks poor (0.72)** because its balanced meta-model pushes probabilities up, so 0.5 is the wrong cut-off for it. We tuned the threshold instead.
- The deployed model (**v2**) is this stacking model retrained with a small set of synthetic modern-scam ads added (see below). The table is from the EMSCAD-only run.
- **kNN looks suspiciously good**: scammers repost the same ad, so kNN partly finds near-copies. That is a leakage risk, not a real skill.
""")

st.markdown("### Choosing the alert threshold")
a, b = st.columns(2)
a.image(str(R / "figures" / "13_v2_pr_curve.png"), caption="Precision-recall curve with the chosen operating point (validation set only).")
b.image(str(R / "figures" / "14_v2_confusion_roc.png"), caption="Confusion matrix and ROC curve on the untouched test set.")
st.markdown("Moving the threshold trades precision for recall. In real use you might prefer higher recall (catch more scams, accept more false alarms).")

st.markdown("### What the model learned")
a, b = st.columns(2)
a.image(str(R / "figures" / "11_lr_top_words.png"), caption="Words that push the logistic regression towards fake vs real (EMSCAD-only model).")
b.image(str(R / "figures" / "12_rf_importance.png"), caption="Most important features in the random forest (EMSCAD-only model).")

with st.expander("Why we added synthetic ads"):
    st.markdown("""
EMSCAD (2012–2014) lacks modern scams: fake-cheque, advance-fee, Telegram interviews, deposit-to-unlock task scams. No public labelled set exists, and the other datasets we found were copies of EMSCAD.
So we generated ads in `src/synth.py`:
- **Fake and legitimate ads from the same generator**, so the model can't learn "generator style = fake". Legitimate ads include hard cases (WhatsApp scheduling, M-Pesa pay, urgent hiring).
- **Only a small set (300 fake, 900 legitimate)** was used. A bigger, 50% fake set hurt real performance (PR-AUC 0.936 → 0.900).
- **Validation and test sets stay 100% real EMSCAD**, so the reported scores are honest.
""")

with st.expander("Avoiding leakage: our checklist"):
    st.markdown("""
- Duplicates were removed **before** splitting (70 / 15 / 15, stratified).
- TF-IDF sits **inside** each pipeline, so it only learns from training data.
- The test set was used **once**, at the end; the threshold was tuned on validation only.
- Known gap: reposted near-copies can still sit in both train and test, which inflates scores. We did not run a grouped split to quantify it.
""")
