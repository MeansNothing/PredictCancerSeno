import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os


class BreastCancerModel:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        model_path = os.path.join(os.path.dirname(__file__), 'breast_cancer_model.pkl')
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.train_model()
            joblib.dump(self.model, model_path)

    def train_model(self):
        # Datos de ejemplo (reemplaza con tu dataset real)
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def predict(self, input_data):
        # input_data debe ser un DataFrame con las mismas columnas que el modelo espera
        prediction = self.model.predict(input_data)
        probabilities = self.model.predict_proba(input_data)
        return prediction, probabilities