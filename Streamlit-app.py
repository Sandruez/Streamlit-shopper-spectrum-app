import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings

# Suppress scikit-learn version mismatch warnings during unpickling
warnings.filterwarnings(
    "ignore",
    message=r"Trying to unpickle estimator .* from version .* when using version .*",
)

# Load Models & Data
try:
    kmeans_model = joblib.load(open("kmeans_rfm_model.joblib", 'rb'))
    scaler = joblib.load(open("rfm_scaler.joblib", "rb"))
    product_similarity_df = joblib.load(open("product_similarity_int_indexed_metrix.joblib", "rb"))
    product_list = list(product_similarity_df.columns)
    similarity_matrix = np.array(product_similarity_df)
except Exception as e:
    st.error("❌ Error loading models or files: " + str(e))
    st.stop()

# --- App Layout ---
st.set_page_config(page_title="Shopper Spectrum", layout="wide")
st.title("🛍️ Shopper Spectrum: Customer & Product Intelligence")

# Tabs: Product Recommender | Customer Segmenter
tab1, tab2 = st.tabs(["📦 Product Recommender", "👥 Customer Segmenter"])

# --- 📦 Product Recommendation ---
with tab1:
    st.header("🔍 Recommend Similar Products")
    st.markdown("Get top-5 similar products using cosine similarity.")

    selected_product = st.selectbox("Choose a Product", product_list)

    if st.button("🧠 Recommend"):
        try:
            index = product_list.index(selected_product)
            similarity_scores = list(enumerate(similarity_matrix[index]))
            top_similar = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]

            with st.expander("🔗 Top 5 Similar Products"):
                for rank, (i, score) in enumerate(top_similar, 1):
                    st.markdown(f"**{rank}.** {product_list[i]}")
        except Exception as e:
            st.error(f"⚠️ Recommendation Error: {str(e)}")

# --- 👥 Customer Segmentation ---
with tab2:
    st.header("🧮 Predict Customer Segment")
    st.markdown("Input customer RFM values to find their segment.")

    col1, col2, col3 = st.columns(3)
    with col1:
        R = st.number_input("📅 Recency (days ago)", min_value=0, max_value=1000, value=100)
    with col2:
        F = st.number_input("🔁 Frequency (orders)", min_value=0, max_value=100, value=5)
    with col3:
        M = st.number_input("💰 Monetary (₹)", min_value=0, max_value=100000, value=2000)

    if st.button("🎯 Predict Segment"):
        try:
            input_df = pd.DataFrame({'Recency': [R], 'Frequency': [F], 'Monetary': [M]})
            scaled_input = scaler.transform(input_df)
            cluster = kmeans_model.predict(scaled_input)[0]

            # Define your own cluster-to-label mapping
            cluster_labels = {
                0: "Regular 🟡",
                1: "Occasional 🟠",
                2: "High-Value 🟢",
                3: "High-Value 🟢"
            }
            st.success(f"✅ **Segment:** {cluster_labels.get(cluster, 'Unknown')} (Cluster {cluster})")
            st.success(f"✅ **This customer belongs to:** {cluster_labels.get(cluster, 'Unknown')} Shopper")

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
