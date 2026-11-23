"""Which words pushed the score up? Reads the text model's coefficients; no extra dependency."""
import html
import re

import numpy as np


def top_words(model, feats, k=10):
    """Top-k terms (unigrams/bigrams) that raised the fraud score, strongest first."""
    lr = model.named_estimators_["lr"]  # the Logistic Regression member: coefficients are readable
    f = lr.named_steps["f"]
    names = f.named_transformers_["tfidf"].get_feature_names_out()
    n = len(names)
    x = f.transform(feats).tocsr()[0, :n]
    contrib = x.multiply(lr.named_steps["m"].coef_[0][:n]).toarray()[0]
    order = np.argsort(-contrib)[:k]
    return [names[i] for i in order if contrib[i] > 0 and "token" not in names[i]]


def highlight(text, terms):
    """HTML-escaped ad text with every word that belongs to a flagged term wrapped in <mark>."""
    words = {w for t in terms for w in t.split()}
    # escape first, then mark (marks add no special characters)
    esc = html.escape(text)
    return re.sub(r"[A-Za-z]+", lambda m: f"<mark>{m.group(0)}</mark>" if m.group(0).lower() in words else m.group(0),
                  esc).replace("\n", "<br>")
