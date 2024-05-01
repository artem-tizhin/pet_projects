import argparse
import pandas as pd
from catboost import CatBoostClassifier

def load_model(path="models/model.cbm"):
    model = CatBoostClassifier()
    model.load_model(path)
    return model

def main(input_path):
    model = load_model()
    data = pd.read_json(input_path)
    preds = model.predict(data)
    probs = model.predict_proba(data)[:, 1]

    for i, (p, prob) in enumerate(zip(preds, probs)):
        print(f"Клиент {i+1}: {'Уйдет' if p == 1 else 'Останется'} (вероятность ухода = {prob:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="JSON-файл с данными клиентов")
    args = parser.parse_args()
    main(args.input)

