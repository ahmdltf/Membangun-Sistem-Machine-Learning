import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    # Mengaktifkan autolog MLflow
    mlflow.set_experiment("Basic Model")
    mlflow.sklearn.autolog()

    # Load dataset preprocessing
    df = pd.read_csv("dataset_preprocessing.csv")

    # Split data
    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Training model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluasi
    model.score(X_test, y_test)

if __name__ == "__main__":
    main()
