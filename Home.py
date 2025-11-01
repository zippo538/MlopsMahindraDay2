import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from config.config import Config
from utils.styling import load_css

st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout=Config.LAYOUT
)
load_css()

st.title("🏠 New York House Price Prediction")
st.markdown("""
Welcome to the New York House Price Prediction App! This application helps you:
- Explore the Boston Housing Dataset
- Understand feature relationships
- Make price predictions using machine learning
- View model performance metrics

Use the navigation menu on the left to explore different sections of the app.
""")

@st.cache_data
def load_data():
    return pd.read_csv(Config.DATA_PATH)

try:
    df = load_data()
    df = df[Config.FEATURE_COLUMN]
    
     # === HEADER ===
    st.title("🏠 Housing Dataset Summary")
    st.markdown("Menampilkan ringkasan umum dari dataset perumahan, termasuk informasi jumlah data, fitur, dan statistik utama.")

    st.divider()
    
    # === METRICS SECTION ===
    st.subheader("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Jumlah Data", f"{df.shape[0]:,}")
    with col2:
        st.metric("🧩 Jumlah Fitur", len(Config.FEATURE_COLUMN))
    with col3:
        st.metric("💰 Rata-rata Harga", f"${df[Config.TARGET_COLUMN].mean():,.2f}")
    with col4:
        st.metric("📉 Harga Minimum", f"${df[Config.TARGET_COLUMN].min():,.2f}")

    st.divider()
    
    # === DISTRIBUSI HARGA ===
    st.subheader("📈 Distribusi Harga Rumah")
    fig = px.histogram(x=df['PRICE'], nbins=30, title="Distribusi Harga Rumah New York", color_discrete_sequence=["#4E79A7"])
    
    fig.update_layout(
        xaxis_title="Harga Rumah",
        yaxis_title="Jumlah",
        title_x=0.5,
        bargap=0.1,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    
    # === FEATURE CORRELATION HEATMAP ===
    st.subheader("🧠 Korelasi Antar Fitur")
    corr = df.corr(numeric_only=True)
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Heatmap Korelasi Fitur",
        aspect="auto"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()
    
    # === FEATURE DESCRIPTIONS ===
    st.subheader("📝 Deskripsi Fitur")
    st.markdown("Berikut adalah keterangan singkat setiap fitur yang digunakan dalam analisis:")

    desc_df = pd.DataFrame.from_dict(
        Config.FEATURE_DESCRIPTIONS,
        orient='index',
        columns=['Deskripsi']
    ).reset_index().rename(columns={'index': 'Fitur'})
    st.dataframe(desc_df, use_container_width=True, hide_index=True)

    st.divider()

    # === SAMPLE DATA ===
    st.subheader("🔍 Contoh Data")
    num_rows = st.slider("Pilih jumlah baris yang ingin ditampilkan:", 5, 20, 5)
    st.dataframe(df.head(num_rows), use_container_width=True)

    st.divider()

    # === RINGKASAN AKHIR ===
    st.success(f"""
    ✅ **Kesimpulan Singkat:**
    - Dataset ini memiliki **{df.shape[0]:,}** baris data dan **{len(Config.FEATURE_COLUMN)}** fitur utama.
    - Rata-rata harga rumah (`PRICE`) adalah sekitar **${df[Config.TARGET_COLUMN].mean():,.2f}**.
    - Fitur yang paling berkorelasi dengan harga rumah dapat dilihat pada heatmap di atas.
    """)

except Exception as e:
    st.error(f"⚠️ Gagal memuat dataset: {str(e)}")