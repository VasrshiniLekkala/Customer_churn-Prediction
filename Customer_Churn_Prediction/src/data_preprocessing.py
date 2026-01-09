import pandas as pd

def preprocess_data(path):
    df = pd.read_csv(path)

    # Drop customerID (not useful for prediction)
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Convert TotalCharges to numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Encode target column
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df
