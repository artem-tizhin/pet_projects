Churn Prediction (Bank Customers)

Pet-проект по машинному обучению: прогнозирование ухода клиента из банка. Модель обучена на датасете Kaggle и позволяет по характеристикам клиента предсказать, уйдёт ли он.

Стек технологий
- Python 3.10
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost

Структура проекта
- `notebooks/` — разведочный анализ данных (EDA)
- `src/` — код для препроцессинга и обучения модели
- `models/` — сохранённая модель
- `tests/` — базовые тесты

Установка
```bash
git clone https://github.com/artem-tizhin/churn-prediction.git
cd churn-prediction
pip install -r requirements.txt

