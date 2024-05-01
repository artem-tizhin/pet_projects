import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path: str = "data/churn.csv"):
    df = pd.read_csv(path)
    return df

def preprocess_data(df: pd.DataFrame):
    # Целевая переменная
    y = df["Exited"]
    X = df.drop(columns=["Exited", "RowNumber", "CustomerId", "Surname"], errors="ignore")

    # Категориальные признаки
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    return X, y, cat_features

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

