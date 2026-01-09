from data_preprocessing import preprocess_data
from src.feature_engineering import feature_engineering
from src.train_model import train_model
from src.evaluate_model import evaluate_model

DATA_PATH = "dataset/telco_customer_churn.csv"

def main():
    df = preprocess_data(DATA_PATH)
    X, y = feature_engineering(df)
    model, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
