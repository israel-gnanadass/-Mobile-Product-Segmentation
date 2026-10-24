from pathlib import Path
import pandas as pd
import numpy as np

NUMERIC_COLUMNS = [
    "age", "price_usd", "exchange_rate_to_usd", "rating",
    "battery_life_rating", "camera_rating", "performance_rating",
    "design_rating", "display_rating", "helpful_votes"
]

def load_and_clean_data(path="data/mobile_reviews_raw.csv"):
    """Load raw reviews, normalize column names/types, remove duplicates and impute missing values."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    if "verified_purchase" in df.columns:
        df["verified_purchase"] = df["verified_purchase"].astype("boolean")
    df = df.drop_duplicates().copy()
    for col in df.columns:
        if col == "review_date":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().any():
                median = df[col].median()
                df[col] = df[col].fillna(median if pd.notna(median) else 0)
        elif str(df[col].dtype) == "boolean":
            df[col] = df[col].fillna(False)
        elif df[col].isna().any():
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "Unknown")
    return df

def product_table(df):
    """Aggregate reviews into one row per mobile model for product-level analysis."""
    required = {"brand", "model"}
    if not required.issubset(df.columns):
        raise ValueError("Dataset must include brand and model columns.")
    agg = {
        "price_usd": "median",
        "rating": "mean",
        "battery_life_rating": "mean",
        "camera_rating": "mean",
        "performance_rating": "mean",
        "design_rating": "mean",
        "display_rating": "mean",
        "helpful_votes": "mean",
        "review_id": "count",
    }
    agg = {k:v for k,v in agg.items() if k in df.columns}
    products = df.groupby(["brand", "model"], dropna=False).agg(agg).reset_index()
    products = products.rename(columns={"review_id":"review_count"})
    return products
