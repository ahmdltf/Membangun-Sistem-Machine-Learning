import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def load_data():
    # Memuat dataset hasil preprocessing
    df = pd.read_csv('dataset_preprocessing.csv')
    return df

def split_data(df):
    # Memisahkan fitur (X) dan target (y)
    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]
    
    # Split data train dan test
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    # Melatih model Random Forest
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def main():
    # Mengaktifkan autolog MLflow
    mlflow.sklearn.autolog()
    
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    
    model = train_model(X_train, y_train)
    
    # Evaluasi model (otomatis dicatat MLflow)
    model.score(X_test, y_test)

if __name__ == "__main__":
    # Menjalankan pipeline modelling
    main()
