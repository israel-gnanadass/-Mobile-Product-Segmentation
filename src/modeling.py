import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

FEATURES = [
    "price_usd", "rating", "battery_life_rating", "camera_rating",
    "performance_rating", "design_rating", "display_rating", "helpful_votes"
]

def fit_segments(products, n_clusters=4, random_state=42):
    """Scale product-level features and fit K-Means clustering."""
    model_data = products.copy()
    features = [c for c in FEATURES if c in model_data.columns]
    if len(model_data) < 2:
        model_data["cluster"] = 0
        return model_data, None, None
    k = max(2, min(int(n_clusters), len(model_data)))
    X = model_data[features].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    model_data["cluster"] = km.fit_predict(X_scaled)
    return model_data, km, scaler

def recommend_similar(products, selected_model, top_n=5):
    """Recommend products using cosine similarity over scaled product attributes."""
    if products.empty:
        return products.copy()
    features = [c for c in FEATURES if c in products.columns]
    X = products[features].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).fillna(0)
    X_scaled = StandardScaler().fit_transform(X)
    sims = cosine_similarity(X_scaled)
    candidates = products.reset_index(drop=True)
    matches = candidates.index[candidates["model"].astype(str) == str(selected_model)].tolist()
    if not matches:
        return candidates.iloc[0:0].copy()
    idx = matches[0]
    scores = sims[idx]
    order = np.argsort(scores)[::-1]
    order = [i for i in order if i != idx][:int(top_n)]
    result = candidates.iloc[order].copy()
    result["similarity_score"] = scores[order]
    return result.sort_values("similarity_score", ascending=False)
