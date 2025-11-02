import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
from config.config import Config
import json
from utils.styling import load_css

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

load_css()


    
def hitung_outlier_iqr(series):
    """Menghitung jumlah outlier menggunakan Rentang Interkuartil (IQR)."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    # Batas atas dan bawah
    batas_bawah = Q1 - 1.5 * IQR
    batas_atas = Q3 + 1.5 * IQR
    
    # Hitung outlier
    outlier_count = ((series < batas_bawah) | (series > batas_atas)).sum()
    
    # Hitung persentase
    total_data = len(series)
    outlier_percent = (outlier_count / total_data) * 100 if total_data > 0 else 0
    
    return outlier_count, outlier_percent, total_data



# Load and cache data with proper cache decorator
@st.cache_data(ttl=3600)
def load_data():
    """Load and cache the dataset"""
    try:
        return pd.read_csv(Config.DATA_PATH)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data(ttl=3600)
def load_model_artifacts():
    """Load and cache model metrics and feature importance"""
    try:
        metrics = None
                
        # Load metrics if available
        if Config.METRICS_PATH.exists():
            with open(Config.METRICS_PATH, 'r') as f:
                metrics = json.load(f)
        
        return metrics
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None

try:
    # Load data and model artifacts
    df = load_data()
    metrics = load_model_artifacts()

    if df is not None:
        df_numeric = df[Config.FEATURE_COLUMN].select_dtypes(include='number')
        feature_columns = df[Config.FEATURE_COLUMN]
        
        
        st.title("📊 Data Analytics & Model Performance")

        # Create tabs
        tab1, tab2 = st.tabs([
            "Data Analysis", 
            "Model Performance"
        ])

        with tab1:
            st.header("Data Distribution Analysis")
            
            feature = st.selectbox(
                "Select Feature",
                Config.FEATURE_COLUMN
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    df, 
                    x=feature,
                    marginal="box",
                    title=f"Distribution of {feature}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                stats = df[feature].describe()
                st.dataframe(stats,use_container_width=True)
            
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("♾️Duplicated Value",feature_columns.duplicated().sum())
                
            
            # # --- HeatMap Correlation --- 
            
            
            # st.subheader(f"Correlation Heatmap")
            # if df_numeric.shape[1] < 2:
            #     st.warning("Anda memerlukan setidaknya dua kolom numerik untuk membuat heatmap")
            # else :
            #     corr_matrix = df_numeric.corr()
            #     st.dataframe(corr_matrix.style.background_gradient(cmap="plasma",axis=None).format("{:.2f}"),use_container_width=True)
            
            
            # --- Hitung outlier ---
            if df[feature].dtypes != "object" : 
                count, percent, total = hitung_outlier_iqr(df[feature])
                delta_str = f"Target < 5.0%"
                if percent > 5.0:
                    delta_color = "inverse" # Merah jika tinggi
                else:
                    delta_color = "normal"  # Hijau jika rendah

                # Tampilkan metrik menggunakan st.metric
                st.subheader(f"Hasil Analisis Outlier untuk Kolom '{feature}'")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label="Jumlah Total Data",
                        value=f"{total} baris",
                        help="Total observasi dalam kolom ini."
                    )

                with col2:
                    st.metric(
                        label="Jumlah Outlier (IQR)",
                        value=f"{count} baris",
                        delta=f"{total - count} Data Normal",
                        delta_color="off", # Matikan warna delta, hanya untuk informasi tambahan
                        help="Jumlah data yang berada di luar batas IQR (Q1 - 1.5*IQR atau Q3 + 1.5*IQR)."
                    )

                with col3:
                    st.metric(
                        label="Persentase Outlier",
                        value=f"{percent:.2f} %",
                        delta=delta_str,
                        delta_color=delta_color,
                        help="Persentase outlier dari total data. Delta menunjukkan status berdasarkan ambang batas 5.0%."
                    )

                st.divider()

                # --- Tampilan Visualisasi (Opsional) ---
                st.subheader("Visualisasi Outlier (Box Plot)")

                # Buat figure Matplotlib secara eksplisit
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(x=df[feature], ax=ax)
                ax.set_title(f"Box Plot untuk {feature}")
                st.pyplot(fig)
            

        
        with tab2:
            st.header("Model Performance")
            st.header("XGBoost Regressor")
            
            if metrics :
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Test R² Score", f"{metrics['R2_SCORE']:.4f}")
                with col2:
                    st.metric("Test RMSE", f"{metrics['RMSE']:,.2f}")
                with col3:
                    st.metric("Test MAE", f"{metrics['MAE']:,.2f}")
                
                              

            else:
                st.warning("Model metrics and feature importance not available. Please train the model first.")

except Exception as e:
    st.error(f"Error in analytics: {str(e)}")
    
