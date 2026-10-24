from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data_processing import load_and_clean_data, product_table
from src.modeling import fit_segments, recommend_similar

st.set_page_config(page_title="Mobile Product Intelligence", page_icon="📱", layout="wide")
st.title("📱 Mobile Product Segmentation & Recommendation System")
st.caption("Review analytics • K-Means segmentation • Similar-product recommendations")

DATA_PATH = ROOT / "data" / "mobile_reviews_raw.csv"
if not DATA_PATH.exists():
    st.error("Dataset not found. Put mobile_reviews_raw.csv in the data/ folder.")
    st.stop()

@st.cache_data
def get_data(path):
    return load_and_clean_data(path)

df = get_data(str(DATA_PATH))
products = product_table(df)
segmented, _, _ = fit_segments(products, n_clusters=4)

with st.sidebar:
    st.header("Filters")
    brand_options = ["All brands"] + sorted(df["brand"].astype(str).unique().tolist())
    selected_brand = st.selectbox("Brand", brand_options)
    country_options = ["All countries"] + sorted(df["country"].astype(str).unique().tolist())
    selected_country = st.selectbox("Country", country_options)
    st.divider()
    st.write("**Dataset**")
    st.write(f"{len(df):,} cleaned reviews")
    st.write(f"{products.shape[0]:,} unique brand/model combinations")

filtered = df.copy()
if selected_brand != "All brands":
    filtered = filtered[filtered["brand"].astype(str) == selected_brand]
if selected_country != "All countries":
    filtered = filtered[filtered["country"].astype(str) == selected_country]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Reviews", f"{len(filtered):,}")
m2.metric("Average rating", f"{filtered['rating'].mean():.2f} / 5")
m3.metric("Brands", f"{filtered['brand'].nunique():,}")
m4.metric("Countries", f"{filtered['country'].nunique():,}")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Segmentation", "Recommendations", "Data explorer"])

with tab1:
    left, right = st.columns(2)
    with left:
        brand_counts = filtered["brand"].value_counts().rename_axis("Brand").reset_index(name="Reviews")
        st.plotly_chart(px.bar(brand_counts, x="Brand", y="Reviews", title="Reviews by brand"), use_container_width=True)
    with right:
        sent = filtered["sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Reviews")
        st.plotly_chart(px.pie(sent, names="Sentiment", values="Reviews", title="Sentiment distribution"), use_container_width=True)
    rating_brand = filtered.groupby("brand", as_index=False)["rating"].mean().sort_values("rating", ascending=False)
    st.plotly_chart(px.bar(rating_brand, x="brand", y="rating", title="Average rating by brand", range_y=[0,5]), use_container_width=True)
    if "review_date" in filtered.columns:
        trend = filtered.dropna(subset=["review_date"]).copy()
        if not trend.empty:
            trend["month"] = trend["review_date"].dt.to_period("M").astype(str)
            monthly = trend.groupby("month", as_index=False).size().rename(columns={"size":"Reviews"})
            st.plotly_chart(px.line(monthly, x="month", y="Reviews", title="Review volume over time"), use_container_width=True)

with tab2:
    st.subheader("Product segments (K-Means)")
    st.write("Products are grouped using price, average rating, and component ratings. Cluster IDs are algorithmic labels; interpret them using the table and chart rather than assuming a fixed meaning.")
    plot_df = segmented.copy()
    if "price_usd" in plot_df and "rating" in plot_df:
        st.plotly_chart(px.scatter(plot_df, x="price_usd", y="rating", color=plot_df["cluster"].astype(str),
                                   hover_name="model", hover_data=["brand", "review_count"],
                                   title="Product clusters: price vs average rating"), use_container_width=True)
    cluster_summary = plot_df.groupby("cluster").agg(
        products=("model","count"),
        median_price_usd=("price_usd","median"),
        avg_rating=("rating","mean"),
        avg_battery=("battery_life_rating","mean"),
        avg_camera=("camera_rating","mean"),
        avg_performance=("performance_rating","mean"),
    ).reset_index()
    st.dataframe(cluster_summary.round(2), use_container_width=True)
    st.download_button("Download segmented products CSV", plot_df.to_csv(index=False).encode("utf-8"),
                       file_name="segmented_mobile_products.csv", mime="text/csv")

with tab3:
    st.subheader("Find similar mobile products")
    product_labels = (products["brand"].astype(str) + " — " + products["model"].astype(str)).tolist()
    label_to_model = dict(zip(product_labels, products["model"].astype(str)))
    selected_label = st.selectbox("Choose a product", product_labels)
    selected_model = label_to_model[selected_label]
    top_n = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
    recs = recommend_similar(products, selected_model, top_n=top_n)
    if recs.empty:
        st.info("No similar products found for this selection.")
    else:
        show_cols = [c for c in ["brand","model","price_usd","rating","review_count","similarity_score"] if c in recs.columns]
        st.dataframe(recs[show_cols].round(3), use_container_width=True)
        st.caption("Similarity is calculated from scaled structured product attributes; it is not a guarantee of personal preference.")

with tab4:
    st.subheader("Review data")
    st.dataframe(filtered.head(1000), use_container_width=True)
    st.download_button("Download filtered reviews CSV", filtered.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_mobile_reviews.csv", mime="text/csv")
