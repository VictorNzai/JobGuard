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
