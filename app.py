import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load('lr.pkl')

# Streamlit app
st.title('HDB Sales price prediction')

# Define input options
towns = ['Tampines', 'Bedok', 'Punggol']
flat_types = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM']
storey_ranges = ['01 TO 03', '04 TO 06', '07 TO 09']

# User inputs
town_selected = st.selectbox('Select Town', towns)
flat_type_selected = st.selectbox('Select Flat Type', flat_types)
storey_range_selected = st.selectbox('Select Storey Range', storey_ranges)
floor_area_selected = st.slider('Select Floor Area (sqm)', min_value=30, max_value=200, value=70)

# Predict button

if st.button('Predict Price'):

    # Prepare input data
    input_data = {
        'town': town_selected,
        'flat_type': flat_type_selected,
        'storey_range': storey_range_selected,
        'floor_area': floor_area_selected
    }
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame({
        'town': [town_selected],
        'flat_type': [flat_type_selected],
        'storey_range': [storey_range_selected],
        'floor_area': [floor_area_selected]
    })
    # One-hot encode categorical variables
    df_input = pd.get_dummies(input_df, columns=['town', 'flat_type', 'storey_range'])

    # df_input = df_input.to_numpy()

    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

    # predict
    y_unseen_pred = model.predict(df_input)[0]
    st.success(f'Predicted Price: ${y_unseen_pred:,.2f}')

st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1350&q=80");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)