# JobGuard: Fake Job Ad Detector, Step-by-Step Project Guide

**ICS3202 Artificial Intelligence: data cleaning → EDA → ensemble ML → web app → presentation**

Put this file in the root of your project folder. Claude Code can then read it: start a session with *"Read PROJECT_GUIDE.md, we're on Step X."*

The code here has been tested on the real dataset (scikit-learn 1.9, pandas 3.0, Streamlit 1.67). You can check your own numbers against the **"What to expect"** notes in each step.

---

## 0. The project in one paragraph

Fake job adverts are a common scam. Victims are asked to pay "registration fees" or send ID copies. We train a model on the **EMSCAD** dataset of 17,880 real job ads, 866 of which are fraudulent (4.8%). The model is a **stacked ensemble**:
- **Naive Bayes** and **Logistic Regression** on the ad text (TF-IDF)
- **Random Forest** and **Gradient Boosting** on the ad's metadata (logo? salary? company profile?)
- a **Logistic Regression meta-model** that combines all four

We then deploy it as a **Streamlit web app**: you paste an ad and get a fraud score, the suspicious words, and warning signs.

**How it maps to the course (CRISP-DM, Unit 4):**

| CRISP-DM stage | Where in this project |
|---|---|
| Business understanding | Step 7, slides 1–2 |
| Data understanding | Step 3 (EDA) |
| Data preparation | Step 2 (cleaning) + `features.py` |
| Modelling | Step 4 |
| Evaluation | Step 5 (+ the Unit 6 bias check) |
| Deployment | Step 6 (Streamlit app) |

---

## 1. Setup (≈15 min)

### 1.1 Get the dataset
- **Kaggle:** search for *"Real / Fake Job Posting Prediction"* (dataset by Shivam Bansal). Download `fake_job_postings.csv`. You need to be logged in.
- **No login (GitHub mirror):** `https://raw.githubusercontent.com/FelixLuciano/Fake-JobPosting-Prediction/main/src/dataset/fake_job_postings.csv` (~50 MB). It's the same file: 17,880 rows × 18 columns, 866 fraudulent.

Credit the dataset in your slides: *EMSCAD, Employment Scam Aegean Dataset, University of the Aegean*.

### 1.2 Folder structure
```
jobguard/
├── PROJECT_GUIDE.md          ← this file
├── README.md
├── requirements.txt
├── .gitignore                ← include: data/  .venv/  __pycache__/  .ipynb_checkpoints/
├── data/
│   └── fake_job_postings.csv
├── notebooks/
│   ├── 01_cleaning_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── features.py           ← feature engineering (shared by notebooks AND app)
│   └── models.py             ← all model pipelines
├── models/
│   └── jobguard.joblib       ← saved model (created in Step 4)
├── reports/figures/          ← save every chart here as PNG for the slides
└── app.py                    ← Streamlit app
```

### 1.3 Environment (VS Code terminal)
```bash
mkdir jobguard && cd jobguard
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud streamlit joblib ipykernel
pip freeze > requirements.txt
```
- In VS Code, install the **Python** and **Jupyter** extensions. When you open a notebook, select the `.venv` kernel.
- ⚠️ **Pin your scikit-learn version** in `requirements.txt`, e.g. `scikit-learn==1.9.1`. A model saved with one version may fail to load in the app under another version.

---

## 2. Data cleaning: `notebooks/01_cleaning_eda.ipynb` (≈45 min)

Facts about the raw data, for your "Data Understanding" slide:

| Column | % missing |
|---|---|
| salary_range | 84.0% |
| department | 64.6% |
| required_education | 45.3% |
| benefits | 40.3% |
| required_experience | 39.4% |
| function | 36.1% |
| industry | 27.4% |
| employment_type | 19.4% |
| company_profile | 18.5% |
| requirements | 15.1% |
| location | 1.9% |

- There are **281 duplicate rows** once you ignore `job_id`.
- Links, emails and phone numbers were anonymised into tokens like `#URL_abc123#`, `#EMAIL_…#` and `#PHONE_…#`.
- Class balance: **17,014 real vs 866 fake** (4.8%). This is heavily imbalanced, so **accuracy is misleading**: a model that always says "real" scores 95%.

### Cleaning steps (document each one in a markdown cell, since these become a slide)
1. **Drop duplicates:** `df.drop_duplicates(subset=[c for c in df.columns if c != "job_id"])`. Do this **before** splitting the data, otherwise copies of the same ad end up in both train and test (leakage).
2. **Drop `job_id`** (it's an identifier) and **`department`** (65% missing and messy).
3. **Text columns:** fill missing values with `""`. **Categorical columns:** fill with `"Unknown"`.
4. **Don't just fill missing values; turn missingness into features.** Scam ads tend to leave fields blank, so create `has_salary`, `has_company_profile`, `has_requirements` and `has_benefits`.
5. **Location:** `"US, NY, New York"` → `country = "US"`.
6. **Text cleaning:** replace the URL/EMAIL/PHONE tokens with words (`urltoken`, `emailtoken`, …), remove digits and punctuation, lowercase.
7. **Extra numeric features:** link/email/phone counts, description length, capital-letter ratio, `!` count.

### `src/features.py` (tested; the app uses this exact function)
```python
import re
import pandas as pd

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
RAW_CAT_COLS = ["employment_type", "required_experience", "required_education", "industry", "function"]
CAT_COLS = RAW_CAT_COLS + ["country"]
BIN_COLS = ["telecommuting", "has_company_logo", "has_questions"]
ENGINEERED_NUM_COLS = ["has_salary", "has_company_profile", "has_requirements", "has_benefits",
                       "n_urls", "n_emails", "n_phones", "desc_len", "caps_ratio", "n_exclaim"]
NUM_COLS = BIN_COLS + ENGINEERED_NUM_COLS

def clean_text(s: str) -> str:
    s = re.sub(r"#(URL|EMAIL|PHONE)_[^#]+#", r" \1TOKEN ", s)   # dataset's hidden tokens
    s = re.sub(r"http\S+|www\.\S+", " URLTOKEN ", s)            # real links typed in the app
    s = re.sub(r"\S+@\S+", " EMAILTOKEN ", s)                   # real emails typed in the app
    s = re.sub(r"[^a-zA-Z ]", " ", s)
    return re.sub(r"\s+", " ", s).lower().strip()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in TEXT_COLS:
        df[c] = df[c].fillna("").astype(str)
    for c in RAW_CAT_COLS:
        df[c] = df[c].fillna("Unknown")
    for c in BIN_COLS:
        df[c] = df[c].fillna(0).astype(int)
    df["country"] = (df["location"].fillna("").astype(str)
                     .str.split(",").str[0].str.strip().replace("", "Unknown"))
    df["has_salary"] = df["salary_range"].notna().astype(int)
    df["has_company_profile"] = (df["company_profile"].str.len() > 0).astype(int)
    df["has_requirements"] = (df["requirements"].str.len() > 0).astype(int)
    df["has_benefits"] = (df["benefits"].str.len() > 0).astype(int)
    raw = df[TEXT_COLS].agg(" ".join, axis=1)
    df["n_urls"] = raw.str.count(r"#URL_|http|www\.")
    df["n_emails"] = raw.str.count(r"#EMAIL_|\S+@\S+")
    df["n_phones"] = raw.str.count(r"#PHONE_")
    df["desc_len"] = df["description"].str.len()
    df["caps_ratio"] = raw.str.count(r"[A-Z]") / (raw.str.len() + 1)
    df["n_exclaim"] = raw.str.count("!")
    df["text"] = raw.map(clean_text)
    return df
```
In the notebooks, import it with `import sys; sys.path.insert(0, "../src"); from features import build_features`.

---

## 3. EDA: same notebook (≈45 min)

Make each of these charts and save it to `reports/figures/` with `plt.savefig(..., dpi=150, bbox_inches="tight")`. Under each chart, write **one sentence describing what you found**.

1. **Class balance bar chart.** It shows the 4.8% fraud rate and is the reason you'll use F1 and PR-AUC.
2. **Missing values, real vs fake:** a grouped bar chart of the % missing per column, split by class. Fake ads skip company profiles far more often.
3. **Fraud rate by `has_company_logo`.** This one is striking: roughly **16% of ads without a logo are fake, vs 2% with a logo**.
4. **Fraud rate by `employment_type`, `required_experience` and `required_education`** (bar charts).
5. **Top 10 countries and top 10 industries by fraud rate.** Only include groups with at least 50 ads, so tiny groups don't top the chart.
6. **Text length distributions** (description length, real vs fake): boxplots or histograms.
7. **Top words / bigrams per class**, with word clouds or a bar chart of the most frequent bigrams in fake vs real ads.
8. **Correlation heatmap** of the numeric and engineered features against `fraudulent`.

Aim for 4–5 strong findings to put on slides.

---

## 4. Modelling: `notebooks/02_modeling.ipynb` (≈1.5 h)

### 4.1 Split three ways (70 / 15 / 15, stratified)
```python
from sklearn.model_selection import train_test_split
df = build_features(df_dedup)
y = df["fraudulent"]; X = df.drop(columns=["fraudulent"])
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.15/0.85, stratify=y_tmp, random_state=42)
```
- **Train** is for fitting the models.
- **Validation** is for comparing models and choosing the alert threshold.
- **Test** is touched **once**, at the very end.
- Always put TF-IDF *inside* a Pipeline, so it only ever learns from the training data and never sees test data.

### 4.2 `src/models.py`: every model is a full Pipeline (tested)
```python
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from features import CAT_COLS, NUM_COLS

def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50_000, sublinear_tf=True, stop_words="english")

def text_features():   # text only; non-negative, so it works with Naive Bayes
    return ColumnTransformer([("tfidf", tfidf(), "text")])

def all_features():    # text + categories + scaled numbers
    return ColumnTransformer([("tfidf", tfidf(), "text"),
                              ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT_COLS),
                              ("num", StandardScaler(), NUM_COLS)])

def meta_features():   # no text, dense; for gradient boosting
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), CAT_COLS),
                              ("num", "passthrough", NUM_COLS)])

def base_models():
    return {
        "Naive Bayes (text)": Pipeline([("f", text_features()), ("m", ComplementNB())]),
        "Logistic Regression": Pipeline([("f", all_features()), ("m", LogisticRegression(max_iter=3000, C=4, class_weight="balanced"))]),
        "kNN (text, cosine)": Pipeline([("f", text_features()), ("m", KNeighborsClassifier(n_neighbors=7, metric="cosine", weights="distance"))]),
        "Decision Tree": Pipeline([("f", all_features()), ("m", DecisionTreeClassifier(max_depth=20, class_weight="balanced", random_state=42))]),
        "Random Forest": Pipeline([("f", all_features()), ("m", RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42))]),
        "Gradient Boosting (meta)": Pipeline([("f", meta_features()), ("m", HistGradientBoostingClassifier(class_weight="balanced", random_state=42))]),
    }

def ensemble_members():
    b = base_models()
    return [("nb", b["Naive Bayes (text)"]), ("lr", b["Logistic Regression"]),
            ("rf", b["Random Forest"]), ("gb", b["Gradient Boosting (meta)"])]

def voting_model():
    return VotingClassifier(ensemble_members(), voting="soft", n_jobs=-1)

def stacking_model():
    return StackingClassifier(estimators=ensemble_members(),
                              final_estimator=LogisticRegression(class_weight="balanced", max_iter=1000),
                              cv=5, stack_method="predict_proba", n_jobs=-1)
```
- **Why stacking with these four members?** The members see different "views" of the ad: NB and LR read the words, while RF and GB read the structure (logo, salary, missing fields). The meta-model learns how much to trust each one.
- **Class imbalance:** this uses `class_weight="balanced"` throughout. Optional extra: try SMOTE from `imbalanced-learn` on one model, and add a slide comparing it with class weights.

### 4.3 Train and compare on the validation set
For every model in `{**base_models(), "Voting (soft)": voting_model(), "Stacking": stacking_model()}`:
1. Fit it on the training set.
2. Score the validation set with `predict_proba`.
3. Record **F1 @ 0.5, precision, recall, PR-AUC (`average_precision_score`) and ROC-AUC**, plus training time.
4. Put everything in one sorted DataFrame. That table is your results slide.

**What to expect** (our test run; Stacking takes about 2 minutes on a laptop):

| Model | F1 @ 0.5 | PR-AUC | ROC-AUC |
|---|---|---|---|
| **Stacking** | 0.725 | **0.952** | **0.997** |
| Voting (soft) | 0.838 | 0.936 | 0.995 |
| Random Forest | 0.811 | 0.921 | 0.993 |
| Logistic Regression | 0.800 | 0.917 | 0.995 |
| kNN (text) | 0.876 | 0.916 | 0.969 |
| Gradient Boosting (meta only) | 0.639 | 0.851 | 0.981 |
| Naive Bayes (text) | 0.484 | 0.557 | 0.906 |
| Decision Tree | 0.627 | 0.525 | 0.919 |

Points to make on the slide:
- **PR-AUC is the headline metric for imbalanced data.** Stacking ranks frauds best.
- **Stacking's F1 at 0.5 is low because 0.5 is the wrong threshold for it.** Its balanced meta-model pushes the probabilities up, so tuning the threshold (4.4) is essential. This is a good insight to explain to the panel.
- **kNN does suspiciously well.** Scammers repost the same ad many times, so kNN is partly finding near-copies. Mention this on the leakage slide (Step 5). For extra credit, try `GroupShuffleSplit` grouped by `description` and show that the scores drop.

### 4.4 Tune the alert threshold (validation set only)
```python
from sklearn.metrics import precision_recall_curve
import numpy as np
p_val = stack.predict_proba(X_val)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, p_val)
f1s = 2 * prec * rec / (prec + rec + 1e-9)
threshold = float(thr[np.argmax(f1s[:-1])])
```
Plot the precision-recall curve and mark the chosen point on it. *What to expect:* a threshold of about **0.97**.

Talking point: in real use you might prefer **higher recall** (catch more scams, accept more false alarms). Show how moving the threshold trades precision against recall.

### 4.5 Final test, once
Run `classification_report(y_test, p_test >= threshold)` and plot the **confusion matrix** and **ROC curve**.

*What to expect on the fraud class:* **precision ≈ 0.93, recall ≈ 0.79, F1 ≈ 0.85**, test PR-AUC ≈ 0.93, ROC-AUC ≈ 0.995.

### 4.6 Save the model
```python
import joblib
joblib.dump({"model": stack, "threshold": threshold}, "../models/jobguard.joblib", compress=3)
```
The file is about 8 MB, which is fine for GitHub.

### 4.7 (Optional, if time) Tune hyperparameters
Use `RandomizedSearchCV` on the Logistic Regression `C` and the TF-IDF `max_features`, with `scoring="average_precision"` and `StratifiedKFold(5)`. Report the before/after scores on one slide.

### 4.8 (Recommended bonus) Logistic Regression from scratch, ≈30 min
This shows the panel you understand what `.fit()` does inside, using the sigmoid, cross-entropy and gradient descent from Unit 5. Train it on the same TF-IDF text features and compare it with scikit-learn's version.
```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class LogisticRegressionScratch:
    def __init__(self, lr=2.0, epochs=800, l2=1e-4, balanced=True):
        self.lr, self.epochs, self.l2, self.balanced = lr, epochs, l2, balanced

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        # class weights: the rare "fake" class counts as much as "real" in total
        sw = (np.where(y == 1, n / (2 * y.sum()), n / (2 * (n - y.sum())))
              if self.balanced else np.ones(n))
        self.losses = []
        for _ in range(self.epochs):
            p = self.sigmoid(X @ self.w + self.b)                      # forward pass
            err = sw * (p - y)                                         # dLoss/dz (cross-entropy)
            self.w -= self.lr * ((X.T @ err) / n + self.l2 * self.w)   # gradient descent + L2
            self.b -= self.lr * err.mean()
            self.losses.append(-np.mean(sw * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))))
        return self

    def predict_proba(self, X):
        return self.sigmoid(X @ self.w + self.b)

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50_000, sublinear_tf=True, stop_words="english")
A_train = vec.fit_transform(X_train["text"]); A_val = vec.transform(X_val["text"])
scratch = LogisticRegressionScratch().fit(A_train, y_train.values)
p = scratch.predict_proba(A_val)
# plot scratch.losses to show the loss falling over the epochs
```
*What to expect (text features only, on a held-out set):*
- **From scratch:** PR-AUC 0.876, F1 0.721. The loss falls from 0.693 to 0.26.
- **scikit-learn:** PR-AUC 0.933, F1 0.861.

The library version wins because it uses a smarter optimiser (L-BFGS) and tuned regularisation. That's a good thing to say on the slide: *"We built it ourselves to understand it, then used the optimised version in production."*

---

## 5. Evaluation extras: what makes this stand out (≈30 min)

1. **Explainability (XAI, Unit 6).** Extract the top 20 words that push towards *fake* and towards *real* from the LR member:
   - Fitted LR member: `stack.named_estimators_["lr"]`
   - Its TF-IDF vectoriser: `.named_steps["f"].named_transformers_["tfidf"]`
   - Its coefficients: `.named_steps["m"].coef_[0][:len(vocab)]`

   Plot them as a diverging bar chart. Also plot `feature_importances_` from the RF member, using the names from `get_feature_names_out()`.
2. **Bias check (Unit 6).** Compare **recall** and **false-positive rate** for US vs non-US ads on the test set:
   ```python
   from sklearn.metrics import confusion_matrix
   res = X_test.assign(y=y_test.values, pred=(p_test >= threshold).astype(int))
   res["region"] = np.where(res["country"] == "US", "US", "Non-US")
   for g, d in res.groupby("region"):
       tn, fp, fn, tp = confusion_matrix(d.y, d.pred, labels=[0, 1]).ravel()
       print(g, len(d), "recall:", round(tp / max(tp + fn, 1), 3), "FPR:", round(fp / max(fp + tn, 1), 3))
   ```
   *What to expect:* recall is about **0.80 for US ads vs 0.72 for non-US ads**. The model misses more scams outside the US. That's **sample bias**: the training data is mostly American ads from 2012–2014. Say this openly, because it's a strong slide for Kenyan users.

   *Top "fake" words in our run:* link urltoken, earn, money, income, apply using. *Top "real" words:* client, experience, growing, companies.
3. **Leakage audit slide:** duplicates were removed before splitting, TF-IDF sits inside the Pipeline, the test set was used once, and scammers' reposted near-copies inflate scores (the kNN observation).
4. **Kenyan reality check.** Collect 3–5 real Kenyan ads: legitimate ones from job boards, and scam ones forwarded on WhatsApp. **Blank out any phone numbers or names**, then run them through the app. This makes for a memorable demo.

---

## 6. The web app: `app.py` (≈45 min)

Inputs:
- text boxes for title, location, company profile, description, requirements, benefits and salary
- checkboxes for logo, screening questions and remote work
- a select box for employment type

Outputs:
1. **Fraud score** and a verdict, using the saved threshold.
2. **"Words that raised suspicion"**, computed from the LR member: TF-IDF of the input × coefficients, keeping the top positive values.
3. **Rule-based warning signs**, labelled as separate from the model: no company profile, no logo, no requirements, contains an email address or link, lots of `!`.
4. A caption on limitations, e.g. *"Trained on 2012–2014 mostly-US ads; treat the score as a warning, not proof."*

Key rules:
- Build a **one-row DataFrame with the same raw columns as the CSV**, set any field the form doesn't ask for to `None`, then call **the same `build_features()`**.
- Load the model with `@st.cache_resource`, so it isn't reloaded on every click.
- Run it with `streamlit run app.py`.

A sample scam to demo: *"Earn $500 weekly from home! No experience needed. Send your CV and ID copy to hr.jobs@gmail.com today!"* Our version scored this 100% fake, flagged the words *earn, data entry, work home*, and raised all five warning signs.

### Deploy (optional, ≈15 min)
1. Push the repo to GitHub. Commit `models/jobguard.joblib`, but keep `data/` out via `.gitignore`.
2. Go to share.streamlit.io → New app → pick the repo → main file `app.py`.
3. Under Advanced settings, choose the same Python version you trained with.

You get a public URL to show on a slide or as a QR code. **Also record a 30-second screen capture of the app working**, in case the venue Wi-Fi fails.

---

## 7. Slides (≈1 h): about 12 slides

1. **Title:** JobGuard: Detecting Fake Job Ads with Ensemble ML, plus team names and student numbers
2. **Problem:** job scams target young job-seekers; what victims lose (fees, ID copies)
3. **Dataset:** EMSCAD, 17,880 ads, 18 columns, 4.8% fake; credit the source
4. **Data cleaning:** the missing-values table, duplicates removed, missingness turned into features, text cleaning
5. **EDA findings (1):** class imbalance, logo vs fraud rate (16% vs 2%)
6. **EDA findings (2):** missing fields per class, top fake-ad words, text length
7. **Approach:** pipeline diagram (Text → TF-IDF → NB / LR; Metadata → RF / GB → meta-LR) and why you used stacking
8. **Results:** model comparison table; explain why PR-AUC and F1 matter more than accuracy here
9. **Threshold and final test:** PR curve with the chosen point, confusion matrix, test scores
9b. **Under the hood (optional):** Logistic Regression from scratch: the loss curve and its scores vs scikit-learn
10. **Explainability:** top fake vs real words, RF feature importance
11. **Bias and leakage:** US vs non-US results, the near-duplicate reposts, limitations (old, US-centric data)
12. **Live demo:** the app with a Kenyan scam example
13. **Conclusion and future work:** retrain on Kenyan ads, a WhatsApp/Telegram bot, a browser extension, and an "inform vs provoke" media meter (Monarch-inspired; credit Brian Mwai's work)

Tip: assign one presenter per section, and keep each slide's text to three short lines.

---

## 8. Prompts for Claude Code (run them in order)

Make sure Claude Code shows you each step, and that you can explain every line. The panel will ask questions.

1. *"Read PROJECT_GUIDE.md. Create the folder structure, `.gitignore`, `requirements.txt` with pinned versions, and `src/features.py` exactly as in Step 2."*
2. *"Create `notebooks/01_cleaning_eda.ipynb` following Steps 2 and 3. Include a markdown explanation before each code cell, save every figure to `reports/figures/`, and write one finding under each chart."*
3. *"Create `src/models.py` from Step 4.2, then `notebooks/02_modeling.ipynb` following Steps 4.1–4.6. Print a sorted comparison table, plot the PR curve with the chosen threshold, the confusion matrix and the ROC curve, and save the model."*
4. *"Add Step 5 to the modelling notebook: LR top words, RF feature importance, the US vs non-US bias check, and a leakage note about near-duplicate ads."*
5. *"Build `app.py` per Step 6 and run it. Then test it with one obviously fake ad and one realistic legitimate ad."*
6. *"Write a README.md explaining the project, how to run it, the results table, and the limitations."*

---

## 9. Timeline for tonight

| Time | Task |
|---|---|
| 20:00 – 20:15 | Setup and dataset (Step 1) |
| 20:15 – 21:45 | Cleaning and EDA (Steps 2–3) |
| 21:45 – 23:15 | Modelling and evaluation (Steps 4–5) |
| 23:15 – 00:00 | App and screen recording (Step 6) |
| 00:00 – 01:00 | Slides and a rehearsal (Step 7) |

## 10. Final checklist
- [ ] Both notebooks run top to bottom with **Restart & Run All**
- [ ] The numbers on the slides match the notebook outputs
- [ ] The app starts from a fresh terminal (`streamlit run app.py`)
- [ ] There's a backup video of the demo
- [ ] Every team member can explain: why PR-AUC and not accuracy; what stacking does; why the threshold isn't 0.5; one limitation
