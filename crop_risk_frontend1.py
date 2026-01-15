# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from crop_risk_backend1 import simulate_cluster_risks, find_top_combinations

st.set_page_config(page_title="Crop Risk Analyzer", layout="wide")

st.title("🌾 Crop Insurance Portfolio Risk Analyzer")

# --- 1. File Upload Section ---
st.sidebar.header("Data Upload")
yield_file = st.sidebar.file_uploader("Upload Yield File (CSV)", type=["csv"])
thresh_file = st.sidebar.file_uploader("Upload Threshold File (CSV)", type=["csv"])

# --- 2. Parameter Inputs ---
n_sims = st.sidebar.number_input("Number of Simulations", 100, 10000, 1000, step=100)
indemnity = st.sidebar.slider("Indemnity Level (%)", 0, 100, 70) / 100

# --- 3. Simulation Logic with Session State ---
if yield_file and thresh_file:
    # We only read the files once to save time
    yield_df = pd.read_csv(yield_file)
    thresh_df = pd.read_csv(thresh_file)

    if st.sidebar.button("▶ Run Simulations"):
        with st.spinner("Running simulations... please wait."):
            # Call your backend
            dist_df, clust_df = simulate_cluster_risks(
                yield_df, thresh_df, n_sims=n_sims, indemnity=indemnity
            )
            
            # CRITICAL: Store results in session_state so they persist
            st.session_state['district_summary'] = dist_df
            st.session_state['cluster_summary'] = clust_df
            st.success("Simulation complete!")

    # --- 4. Display Results (Only if they exist in state) ---
    if 'cluster_summary' in st.session_state:
        cluster_summary = st.session_state['cluster_summary']
        district_summary = st.session_state['district_summary']

        # Tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📊 Main Summaries", "⚖️ Cluster Comparison", "🏆 Stability Analysis"])

        with tab1:
            st.subheader("Cluster-Level Summary")
            st.dataframe(cluster_summary)
            
            st.subheader("District-Level Summary")
            st.dataframe(district_summary)

        with tab2:
            st.subheader("Compare Two Clusters")
            all_clusters = cluster_summary["Cluster"].unique().tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                c1 = st.selectbox("Select Cluster A", all_clusters, index=0)
            with col2:
                c2 = st.selectbox("Select Cluster B", all_clusters, index=min(1, len(all_clusters)-1))

            if c1 != c2:
                # Filter for the two selected clusters
                comparison_df = cluster_summary[cluster_summary["Cluster"].isin([c1, c2])]
                
                # Show Data
                st.write(f"Data Comparison for {c1} and {c2}")
                st.dataframe(comparison_df)

                # Show Chart - Ensure column names match your backend exactly (lower_case)
                # We use burn_rate and burn_rate_sd
                st.write("#### Burn Rate vs Volatility (SD)")
                chart_data = comparison_df.set_index("Cluster")[["burn_rate", "burn_rate_sd"]]
                st.bar_chart(chart_data)
                
                # Optional: Show Tail Risk Comparison
                st.write("#### Tail Risk (P90 vs P95)")
                tail_data = comparison_df.set_index("Cluster")[["p90", "p95"]]
                st.line_chart(tail_data)
            else:
                st.warning("Please select two different clusters to compare.")

        with tab3:
            st.write("### 🔎 Top Cluster Combinations (Most Stable)")
            combo_df = find_top_combinations(cluster_summary)
            st.dataframe(combo_df)

else:
    st.info("👋 Please upload both Yield and Threshold files in the sidebar to begin.")

# --- 5. Reset Logic ---
if st.sidebar.button("🔁 Clear All Data"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()