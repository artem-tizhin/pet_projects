import joblib
from catboost import CatBoostClassifier
from preprocess import load_data, preprocess_data, split_data

def train_model():
    df = load_data()
    X, y, cat_features = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=100,
        random_seed=42
    )

    model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test))

    model.save_model("models/model.cbm")
    print("Модель сохранена в models/model.cbm")

if __name__ == "__main__":
    train_model()

