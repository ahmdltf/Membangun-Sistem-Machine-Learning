import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data():
    # Memuat dataset hasil preprocessing
    df = pd.read_csv("dataset_preprocessing.csv")
    return df

def split_data(df):
    # Memisahkan fitur dan target
    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def tuning_model(X_train, y_train):
    # Hyperparameter tuning Random Forest
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10]
    }

    model = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(model, param_grid, cv=3)
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_

def evaluate_model(model, X_test, y_test):
    # Evaluasi model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return acc, cm, report

def save_confusion_matrix(cm):
    # Menyimpan confusion matrix sebagai gambar
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

def save_classification_report(report):
    # Menyimpan classification report sebagai file teks
    with open("classification_report.txt", "w") as f:
        f.write(report)

def main():
    # Sinkronkan dengan MLflow UI
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Set nama experiment
    mlflow.set_experiment("Advance Model")

    with mlflow.start_run():

        # Load & split data
        df = load_data()
        X_train, X_test, y_train, y_test = split_data(df)

        # Tuning model
        model, params = tuning_model(X_train, y_train)

        # Evaluasi
        acc, cm, report = evaluate_model(model, X_test, y_test)

        # Logging parameter
        mlflow.log_params(params)

        # Logging metric
        mlflow.log_metric("accuracy", acc)

        # Logging model
        mlflow.sklearn.log_model(model, "model")

        # Simpan artefak tambahan
        save_confusion_matrix(cm)
        save_classification_report(report)

        # Upload artefak ke MLflow
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact("classification_report.txt")

        # Tambahan tag (biar advance makin kuat)
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("author", "Ahmad Latif")

if __name__ == "__main__":
    main()
