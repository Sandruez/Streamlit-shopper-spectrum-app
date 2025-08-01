import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load assets
kmeans_model = joblib.load(open("kmeans_rfm_model.joblib",'rb'))
scaler = joblib.load(open("rfm_scaler.joblib", "rb"))
product_similarity_metrix_df = joblib.load(open("product_similarity_int_indexed_metrix.joblib", "rb"))
product_list = list(product_similarity_metrix_df.columns)
product_similarity_metrix=np.array(product_similarity_metrix_df)

# App title
st.title("🛍️ Customer & Product Intelligence App")

# Tabs for 2 sections
tab1, tab2 = st.tabs(["📦 Product Recommendation", "👤 Customer Segmentation"])

# -------- Product Recommendation ------------
with tab1:
    st.header("🔍 Find Similar Products")
    product_input = st.selectbox("Choose a Product", product_list)

    if st.button("Recommend"):
        try:
            index = product_list.index(product_input)
            similarity_scores = list(enumerate(product_similarity_metrix[index]))
            similar_products = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]
            st.success("Top 5 similar products:")
            for i, (idx, score) in enumerate(similar_products):
                st.write(f"{i + 1}. {product_list[idx]} (Score: {round(score, 2)})")
        except:
            st.error("Product not found or similarity matrix mismatch.")

# -------- Customer Segmentation ------------
with tab2:
    st.header("👤 Predict Customer Segment")
    R = st.number_input("Recency (days)", min_value=0, max_value=1000, value=100)
    F = st.number_input("Frequency (purchases)", min_value=0, max_value=100, value=5)
    M = st.number_input("Monetary (₹)", min_value=0, max_value=100000, value=2000)
    input_df=pd.DataFrame({'Recency':R,
                           'Frequency':F,
                           'Monetary':M
                          },index=[0]
                          )

    if st.button("Predict Segment"):
        input_data = scaler.transform(input_df)
        cluster = kmeans_model.predict(input_data)[0]

        # Cluster to segment mapping (example)
        cluster_map = {
            0: "Regular",
            1: "Occasional",
            2: "High-Value",
            3: "High-Value"
        }
        segment = cluster_map.get(cluster, "Unknown")
        st.success(f"Predicted Segment: **{segment}** (Cluster {cluster})")

