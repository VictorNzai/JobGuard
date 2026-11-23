# JobGuard: Fake Job Ad Detector

ICS3202 Artificial Intelligence project. Paste a job ad into a web app and get a fraud score, the words that raised suspicion, and rule-based warning signs.

**Data:** EMSCAD (Employment Scam Aegean Dataset, University of the Aegean): 17,880 ads, 866 fraudulent (4.8%).

## Approach

A stacked ensemble. Each member sees a different view of the ad:

- **Naive Bayes** and **Logistic Regression** read the text (TF-IDF)
- **Random Forest** and **Gradient Boosting** read the metadata (logo, salary, missing fields)
- A **Logistic Regression meta-model** combines the four

**Deployed model (v2)** is this stack retrained with a small set of synthetic modern-scam ads (300 fake, 900 legitimate, `src/synth.py`) added to the training split only. The validation and test sets are real EMSCAD. v1 (`models/jobguard_v1.joblib`) is the EMSCAD-only model from the notebook.

Duplicates are removed before splitting (70/15/15, stratified), TF-IDF lives inside the pipelines, the test set was used once, and the alert threshold (v2: 0.906) was tuned on the validation set.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Paste a job-posting link (public pages only) or the ad text, or use the demo buttons. The data file is not in the repo; download `fake_job_postings.csv` into `data/` only if you want to re-run the notebooks.

Notebooks, in order: `notebooks/01_cleaning_eda.ipynb`, then `notebooks/02_modeling.ipynb` (this writes `models/jobguard.joblib`).

scikit-learn is pinned to 1.9.1; the saved model may not load under other versions.

## Results

Validation set (see `reports/model_comparison.csv`):

| Model | F1 @ 0.5 | PR-AUC | ROC-AUC |
|---|---|---|---|
| Stacking | 0.718 | 0.951 | 0.996 |
| Voting (soft) | 0.839 | 0.933 | 0.995 |
| Random Forest | 0.827 | 0.925 | 0.993 |
| kNN (text, cosine) | 0.881 | 0.917 | 0.969 |
| Logistic Regression | 0.796 | 0.916 | 0.995 |
| Gradient Boosting (meta) | 0.602 | 0.842 | 0.978 |
| Naive Bayes (text) | 0.484 | 0.557 | 0.906 |
| Decision Tree | 0.622 | 0.491 | 0.918 |

PR-AUC is the headline metric because only 4.8% of ads are fake, so accuracy is misleading. Stacking's F1 at 0.5 is low because its balanced meta-model pushes probabilities up; with the tuned threshold the picture changes.

Final test set (2,640 real ads, v2 at threshold 0.906), fake class: **precision 0.90, recall 0.79, F1 0.84**; PR-AUC 0.925, ROC-AUC 0.994. (EMSCAD-only v1: precision 0.93, recall 0.78, F1 0.85, PR-AUC 0.936.)

Outside check on modern scams (`reports/augmentation_results_300_900.json`): on a 47-ad set (19 fake) v2 caught 53% of fakes at its threshold vs 0% for v1, with no false alarms. That set is tiny, so this is encouraging, not proven. Other public sets we found (`balanced_en`, `adaption`) are copies of EMSCAD, and the Qatar extended dataset has no downloadable data, so none were used. Synthetic-only scores are not reported because the same generator wrote both sets.

## How the app explains a score

The score comes from the stacked model. The explanation is a list of named red flags (`src/redflags.py`) in two tiers:
- **Backed by the dataset:** each flag's fake-rate is measured on EMSCAD, counting each reposted scam template once so repeats don't inflate it (e.g. "pay amount + no experience / work from home": 25% fake vs 4.5% typical).
- **Common scam tricks:** fee requests, M-Pesa/Paybill, WhatsApp-only contact, personal email, ID/bank details, overseas placement, urgency. These come from consumer-protection guidance; EMSCAD cannot measure them (emails and phones were anonymised and it predates these scams), so no rate is shown.

Also found: a high salary is *not* a signal in EMSCAD (median top-of-range 45k for fake vs 50k for real), so the app does not flag salary size.

## Limitations

- **Old, US-centric data (2012-2014).** Recall is 0.79 for US ads but 0.72 for non-US ads, so the model misses more scams outside the US. This is sample bias.
- **Near-duplicate reposts** inflate scores, kNN especially. Scammers repost the same ad, so some test ads have near-copies in training. A grouped split was not run.
- **Not tested on Kenyan ads.** Treat the score as a warning, not proof.

## Layout

```
app.py              Streamlit entry point + navigation
views/              pages: home, check a job ad, the data, how we built it, findings
src/features.py     feature engineering (shared by notebooks and app)
src/models.py       all model pipelines
src/redflags.py     named red flags + their measured fake-rates (python src/redflags.py rewrites reports/red_flag_stats.json)
src/synth.py        synthetic fake + legit ads (training only)
src/augment_experiment.py  EMSCAD-only vs +synthetic, on real and outside data
src/v2_report.py    final v2 numbers and figures
src/summary.py      precomputes reports/dataset_summary.json for the data page (python src/summary.py)
src/fetch.py        reads a job-posting URL (schema.org JobPosting data, else page text)
notebooks/          cleaning + EDA, modelling + evaluation
models/             saved model
reports/figures/    charts used in the slides
```
