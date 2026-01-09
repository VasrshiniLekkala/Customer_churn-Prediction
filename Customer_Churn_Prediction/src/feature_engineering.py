from sklearn.preprocessing import LabelEncoder

def feature_engineering(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    return X, y

