import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ],
        remainder="drop"
    )

    return preprocessor


def preprocess_classification():
    df = pd.read_csv("ingested/B.csv")

    X = df.drop(columns=["placement_status", "salary_package_lpa"])
    y = df["placement_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_data = X_train.copy()
    train_data["placement_status"] = y_train
    train_data = train_data.drop_duplicates()

    X_train = train_data.drop("placement_status", axis=1)
    y_train = train_data["placement_status"]

    preprocessor = build_preprocessor(X_train)

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    os.makedirs("artifact", exist_ok=True)
    joblib.dump(preprocessor, "artifact/preprocessor_clf.pkl")

    train = pd.DataFrame(
        X_train_p.toarray() if hasattr(X_train_p, "toarray") else X_train_p
    )
    train["placement_status"] = y_train.reset_index(drop=True)

    test = pd.DataFrame(
        X_test_p.toarray() if hasattr(X_test_p, "toarray") else X_test_p
    )
    test["placement_status"] = y_test.reset_index(drop=True)

    train.to_csv("train_clf.csv", index=False)
    test.to_csv("test_clf.csv", index=False)

    print("Classification preprocessing completed")


def preprocess_regression():
    df = pd.read_csv("ingested/B.csv")

    X = df.drop(columns=["salary_package_lpa", "placement_status"])
    y = df["salary_package_lpa"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_data = X_train.copy()
    train_data["salary_package_lpa"] = y_train
    train_data = train_data.drop_duplicates()

    X_train = train_data.drop("salary_package_lpa", axis=1)
    y_train = train_data["salary_package_lpa"]

    preprocessor = build_preprocessor(X_train)

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    os.makedirs("artifact", exist_ok=True)
    joblib.dump(preprocessor, "artifact/preprocessor_reg.pkl")

    train = pd.DataFrame(
        X_train_p.toarray() if hasattr(X_train_p, "toarray") else X_train_p
    )
    train["salary_package_lpa"] = y_train.reset_index(drop=True)

    test = pd.DataFrame(
        X_test_p.toarray() if hasattr(X_test_p, "toarray") else X_test_p
    )
    test["salary_package_lpa"] = y_test.reset_index(drop=True)

    train.to_csv("train_reg.csv", index=False)
    test.to_csv("test_reg.csv", index=False)

    print("Regression preprocessing completed")


if __name__ == "__main__":
    preprocess_classification()
    preprocess_regression()