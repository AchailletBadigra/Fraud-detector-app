import pandas as pd
import pickle


api_url = "http://localhost:8502"

import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.title("Fraud Prediction with File Upload")

# Display site header
#image = Image.open("../images/dsif header.jpeg")

image_path = "../images/dsif header 2.jpg"
try:
    # Open and display the image
    img = Image.open(image_path)
    st.image(img, use_column_width=True)  # Caption and resizing options
except FileNotFoundError:
    st.error(f"Image not found at {image_path}. Please check the file path.")

transaction_amount = st.number_input("Transaction Amount")
customer_age = st.number_input("Customer Age")
customer_balance = st.number_input("Customer Balance")

data = {
        "transaction_amount": transaction_amount,
        "customer_age": customer_age,
        "customer_balance": customer_balance
    }

if st.button("Show Feature Importance"):
    import matplotlib.pyplot as plt
    response = requests.get(f"{api_url}/feature-importance")
    feature_importance = response.json().get('feature_importance', {})

    features = list(feature_importance.keys())
    importance = list(feature_importance.values())

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    st.pyplot(fig)

if st.button("Predict and show prediction confidence"):
    # Make the API call

    response = requests.post(f"{api_url}/predict/",
                            json=data)
    result = response.json()
    confidence = result['confidence']

    if result['fraud_prediction'] == 0:
        st.write("Prediction: Not fraudulent")
    else:
        st.write("Prediction: Fraudulent")

     # Confidence Interval Visualization
    labels = ['Not Fraudulent', 'Fraudulent']
    fig, ax = plt.subplots()
    ax.bar(labels, confidence, color=['green', 'red'])
    ax.set_ylabel('Confidence')
    ax.set_title('Prediction Confidence')
    st.pyplot(fig)

if st.button("Predict and show SHAP values"):
    response = requests.post(f"{api_url}/predict/",
                             json=data)
    result = response.json()

    if result['fraud_prediction'] == 0:
        st.write("Prediction: Not fraudulent")
    else:
        st.write("Prediction: Fraudulent")

     ######### SHAP #########
     # Extract SHAP values and feature names
    shap_values = np.array(result['shap_values'])
    features = result['features']

    # Display SHAP values
    st.subheader("SHAP Values Explanation")

     # Bar plot for SHAP values
    fig, ax = plt.subplots()
    ax.barh(features, shap_values[0])
    ax.set_xlabel('SHAP Value (Impact on Model Output)')
    st.pyplot(fig)


# Section for uploading a CSV file
st.header("Upload a CSV file with transaction data")

uploaded_file = st.file_uploader("Upload a CSV file with transaction data", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("File uploaded successfully! Here's a preview:")
    st.write(df.head())

    # Load the pre-trained model (assuming it's saved as fraud_model.joblib in the models folder)
    from joblib import load

    with open('../models/lr1-pipeline.pkl', 'rb') as f:
        model = pickle.load(f)

    # Add fraud predictions
    # df['fraud_prediction'] = model.predict(df[['transaction_amount', 'transaction_time', 'customer_age', 'customer_balance']])
    df['fraud_prediction'] = model.predict(df[['transaction_amount', 'customer_age', 'customer_balance']])

    st.write("Predictions added to the dataset:")
    st.write(df.head())

    # Download predictions as CSV
    st.download_button(
        label="Download predictions as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='fraud_predictions.csv',
        mime='text/csv'
    )

    # Add feature for transaction amount to balance ratio
    df['amount_balance_ratio'] = df['transaction_amount'] / df['customer_balance']
    st.write("New feature 'amount_balance_ratio' added to the dataset.")

    # Interactive scatter plot
    columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    x_axis = st.selectbox('Select X-axis', columns)
    y_axis = st.selectbox('Select Y-axis', columns)

    if x_axis and y_axis:
        fig, ax = plt.subplots()
        ax.scatter(df[x_axis], df[y_axis])
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f'Scatter Plot: {x_axis} vs {y_axis}')
        st.pyplot(fig)
