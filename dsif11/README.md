
# Batch Fraud Prediction with File Upload

## Overview

This project is a **Streamlit** web application that allows users to upload a CSV file with transaction data and predicts the likelihood of fraud for each transaction using a pre-trained machine learning model. The predictions can be downloaded as a CSV file. The app also provides interactive data visualizations to help users explore the relationship between different transaction variables.

## Features

- **File Upload**: Upload a CSV file containing transaction data (e.g., `transaction_amount`, `transaction_time`, `customer_age`, `customer_balance`).
- **Fraud Prediction**: Predict the likelihood of fraud for each transaction using a pre-trained machine learning model.
- **CSV Download**: Download the predictions and additional features (e.g., transaction-to-balance ratio) as a CSV file.
- **Interactive Visualizations**: Generate scatter plots to explore relationships between different variables, including the custom feature **transaction amount to balance ratio**.

## Requirements

To run this project locally, you'll need the following:
- Python 3.7 or higher
- Required Python packages:
  - `streamlit`
  - `fastapi`
  - `uvicorn`
  - `pandas`
  - `joblib`
  - `scikit-learn`
  - `shap`
  - `requests`

## Setup Instructions

### 1. Clone the Repository
First, clone the project repository to your local machine.

```bash
git clone <repository-url>
```

### 2. Install Dependencies
Navigate to the project directory and install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` file, install the packages manually:

```bash
pip install streamlit fastapi uvicorn pandas joblib scikit-learn shap requests
```

### 3. Run the FastAPI Backend (Fraud Prediction API)
Start the **FastAPI** server to handle fraud prediction requests. Open a terminal and run:

```bash
uvicorn dsif11app-fraud:app --reload --port 8502
```

This will start the API server on `localhost:8502`.

### 4. Run the Streamlit App
In a separate terminal, start the **Streamlit** app to open the web interface:

```bash
streamlit run app-v4/src/dsif11app-fraud-streamlit.py
```

The Streamlit app will be available at `localhost:8501`.

## How to Use the App

1. **Upload a CSV File**: In the web interface, upload a CSV file containing transaction data. The CSV file should have the following columns:
   - `transaction_amount`
   - `transaction_time`
   - `customer_age`
   - `customer_balance`

   Example CSV format:

   ```csv
   transaction_amount,transaction_time,customer_age,customer_balance
   100.5,5,34,5000.0
   200.0,12,45,2500.0
   320.45,15,25,1000.0
   ```

2. **Make Predictions**: After uploading the file, the app will automatically run fraud predictions for each transaction and display the results in a table.

3. **Download Predictions**: After predictions are made, you can download the results (including the fraud prediction column) as a CSV file.

4. **Interactive Scatter Plot**: Use the scatter plot feature to select and visualize relationships between different columns in the dataset, including the custom feature **transaction amount to balance ratio**.

## Example CSV File

Here is an example of the format for the CSV file you can upload:

```csv
transaction_amount,transaction_time,customer_age,customer_balance
100.5,5,34,5000.0
200.0,12,45,2500.0
320.45,15,25,1000.0
450.75,7,29,3000.0
...
```

## Troubleshooting

- Ensure that both the **FastAPI** server and **Streamlit** app are running simultaneously.
- If you encounter file path issues when loading the model, ensure that the model file (`lr1-pipeline.pkl`) is correctly located in the `app-v4/models/` directory.
- Make sure the CSV file you're uploading has the correct format and includes the necessary columns (`transaction_amount`, `customer_age`, `customer_balance`).

## License

This project is licensed under the **MIT License**.
