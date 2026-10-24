from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data_processing import load_and_clean_data, product_table
from src.modeling import fit_segments

data = load_and_clean_data(ROOT / "data" / "mobile_reviews_raw.csv")
products = product_table(data)
segmented, _, _ = fit_segments(products, 4)
out = ROOT / "outputs"
out.mkdir(exist_ok=True)
data.to_csv(out / "mobile_reviews_cleaned.csv", index=False)
segmented.to_csv(out / "segmented_products.csv", index=False)
data.describe(include="all").transpose().to_csv(out / "data_profile.csv")
summary = segmented.groupby("cluster").agg(
    products=("model","count"), median_price_usd=("price_usd","median"),
    average_rating=("rating","mean"), average_battery=("battery_life_rating","mean"),
    average_camera=("camera_rating","mean"), average_performance=("performance_rating","mean")
).round(2)
summary.to_csv(out / "cluster_summary.csv")
print(f"Done. Cleaned {len(data):,} reviews and segmented {len(products):,} products.")
print(f"Outputs saved to: {out}")
print(summary)
