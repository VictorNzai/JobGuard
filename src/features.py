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
