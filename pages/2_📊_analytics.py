import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from config.config import Config
import json
from utils.styling import load_css

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

load_css()

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
                st.dataframe(stats)

        

        with tab2:
            st.header("Model Performance")
            
            if metrics :
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Test R² Score", f"{metrics['R2_SCORE']:.4f}")
                with col2:
                    st.metric("Test RMSE", f"${metrics['RMSE']:,.2f}")
                with col3:
                    st.metric("Test MAE", f"${metrics['MAE']:,.2f}")
                
                
                # Feature Importance Section
                st.subheader("Feature Importance Analysis")
                
                # Create DataFrame from feature importance
                importance_df = pd.DataFrame({
                    'Feature': list(feature_importance.keys()),
                    'Importance': list(feature_importance.values())
                }).sort_values('Importance', ascending=True)

                # Horizontal bar chart
                fig = go.Figure(go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker=dict(
                        color='rgb(26, 118, 255)',
                        line=dict(color='rgba(26, 118, 255, 1.0)', width=1)
                    )
                ))

                fig.update_layout(
                    title='Feature Importance',
                    xaxis_title='Importance Score',
                    yaxis_title='Features',
                    template='plotly_white',
                    height=400,
                    margin=dict(l=0, r=0, t=30, b=0)
                )

                st.plotly_chart(fig, use_container_width=True)

                # Feature Importance Details
                with st.expander("Feature Importance Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Top Features")
                        top_features = importance_df.tail(3).copy()
                        top_features['Importance (%)'] = top_features['Importance'] * 100
                        st.dataframe(
                            top_features[['Feature', 'Importance (%)']].round(2)
                        )
                    
                    with col2:
                        st.markdown("### Feature Importance Distribution")
                        fig = px.pie(
                            importance_df,
                            values='Importance',
                            names='Feature',
                            title='Feature Importance Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("Model metrics and feature importance not available. Please train the model first.")

except Exception as e:
    st.error(f"Error in analytics: {str(e)}")