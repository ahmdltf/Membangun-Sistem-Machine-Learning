import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    # Memproses data numerik menjadi list
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("CI Model")

    df = pd.read_csv("dataset_preprocessing.csv")

    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    with mlflow.start_run():
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        # Simpan artifact
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d')
        plt.savefig("confusion_matrix.png")

        with open("report.txt", "w") as f:
            f.write(report)

        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact("report.txt")

if __name__ == "__main__":
    main()
