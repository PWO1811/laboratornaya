import sys
import os
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import joblib


from src.preprocessing import load_data, create_preprocessor


def train_model():

    data = load_data(os.path.join('data', 'Laptop_price.csv'))
    X = data.drop('Price', axis=1)
    y = data['Price']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    preprocessor = create_preprocessor()
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])


    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE: {mae}")


    os.makedirs('models', exist_ok=True)
    joblib.dump(model, os.path.join('models', 'model.joblib'))

    return model


if __name__ == "__main__":
    train_model()