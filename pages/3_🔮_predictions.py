import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from config.config import Config
from utils.styling import load_css

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")

# Load CSS
load_css()

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# Prediction Form
with st.form("prediction_form"):
    st.subheader("Enter House Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        price = st.number_input(
            "Price $",
            min_value=1000,
            max_value=100000000,
            value=1100,
            help=Config.FEATURE_DESCRIPTIONS['PRICE']
        )
        
        beds = st.slider(
            "Number of Beds ",
            min_value=1,
            max_value=10,
            value=2,
            help=Config.FEATURE_DESCRIPTIONS['BEDS']
        )
        
    
    with col2:
        bath = st.slider(
            "Number of Bath",
            min_value=1,
            max_value=5,
            value=2,
            help=Config.FEATURE_DESCRIPTIONS['BATH']
        )
        
        propertysqft = st.slider(
            "House Area",
            min_value=100,
            max_value=10000,
            value=200,
            help=Config.FEATURE_DESCRIPTIONS['PROPERTYSQFT']
        )
    with col3 : 
        locality = st.selectbox(
            "Choose Your Domain",
            ['New York', 'New York County', 'The Bronx', 'Kings County',
       'Bronx County', 'Queens County', 'Richmond County',
       'United States', 'Brooklyn', 'Queens', 'Flatbush'],
            help=Config.FEATURE_DESCRIPTIONS['LOCALITY']
        )
        
       
    
    submitted = st.form_submit_button("🏠 Predict Price")

if submitted:
    input_data = {
        "PRICE": price,
        "BEDS": beds,
        "BATH": bath,
        "PROPERTYSQFT": propertysqft,
        "LOCALITY": locality,
    }

# Update API endpoint URL untuk Docker
# API_URL = "http://fastapi:8000"  # Gunakan nama service dari docker-compose
    
    try:
        with st.spinner('Making prediction...'):
            response = requests.post(
                "http://localhost:8000/predict",
                json=input_data
            )
            
            #--- Jika pake docker ---
            # # Update prediction request
            # response = requests.post(
            #     f"{API_URL}/predict",
            #     json=input_data
            # )
            
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                
                # Store prediction
                st.session_state.predictions.append({
                    "prediction": prediction,
                    **input_data
                })
                
                st.success(f"### Predicted House Price New York : ${prediction:,.2f}")
                
                # Display feature values
                st.subheader("Feature Values Used")
                feature_df = pd.DataFrame([input_data]).T
                feature_df.columns = ['Value']
                st.dataframe(feature_df)
                
            else:
                st.error(f"Error making prediction: {response.text}")
                
    except requests.exceptions.ConnectionError:
        st.error("Error connecting to the prediction service. Please make sure the API is running.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Display prediction history
if st.session_state.predictions:
    st.header("Prediction History")
    
    df_pred = pd.DataFrame(st.session_state.predictions)
    
        
    
    st.subheader("Recent Predictions")
    for idx, pred in enumerate(df_pred.tail(5).iloc[::-1].to_dict('records')):
            with st.expander(f"Prediction {len(df_pred) - idx}", expanded=idx == 0):
                cols = st.columns(5)
                with cols[0]:
                    st.metric("Price", f"${pred['prediction']:,.2f}")
                with cols[1]:
                    st.metric("Beds", f"{pred['BEDS']:}")
                with cols[2]:
                    st.metric("Bath", f"{pred['BATH']:}")
                with cols[3]:
                    st.metric("House Area", f"{pred['PROPERTYSQFT']}")
                with cols[4]:
                    st.metric("Domain", f"{pred['LOCALITY']}")
    
    if st.button("Clear Prediction History"):
            st.session_state.predictions = []
            st.experimental_rerun()
    
    # Visualization section
    st.header("Prediction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter plot of predictions vs rooms
        fig1 = px.scatter(
            df_pred,
            x='BEDS',
            y='prediction',
            title='Predicted Price vs Number of Rooms',
            labels={
                'BEDS': 'Number of beds',
                'prediction': 'Predicted Price ($)'
            }
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Create manual BEDS ranges for grouping
        df_pred['Room_Range'] = pd.cut(
            df_pred['BEDS'],
            bins=[2, 4, 5, 6, 7, 8, 9],
            labels=['2-4', '4-5', '5-6', '6-7', '7-8', '8-9']
        )
        
        fig2 = px.box(
            df_pred,
            x='Room_Range',
            y='prediction',
            title='Price Distribution by Room Ranges',
            labels={
                'Room_Range': 'Room Ranges',
                'prediction': 'Predicted Price ($)'
            }
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Statistics
    st.subheader("Prediction Statistics")
    stats_cols = st.columns(4)
    
    with stats_cols[0]:
        st.metric("Average Price", f"${df_pred['prediction'].mean():,.2f}")
    with stats_cols[1]:
        st.metric("Highest Price", f"${df_pred['prediction'].max():,.2f}")
    with stats_cols[2]:
        st.metric("Lowest Price", f"${df_pred['prediction'].min():,.2f}")
    with stats_cols[3]:
        st.metric("Total Predictions", len(df_pred))
    
    # Download predictions
    if not df_pred.empty:
        st.download_button(
            label="Download Prediction History",
            data=df_pred.to_csv(index=False).encode('utf-8'),
            file_name="house_price_predictions.csv",
            mime="text/csv"
        )

else:
    st.info("No predictions made yet. Use the form above to make predictions.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)