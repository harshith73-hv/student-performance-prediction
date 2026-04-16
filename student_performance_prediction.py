"""
Student Performance Prediction using Machine Learning

What this script does:
1. Creates a small sample student dataset
2. Preprocesses the data
3. Trains two regression models
4. Evaluates model performance
5. Prints the better model
6. Saves a plot of actual vs predicted scores

Requirements:
pip install pandas numpy scikit-learn matplotlib seaborn
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_sample_dataset() -> pd.DataFrame:
    """Create a simple sample dataset for student performance prediction."""
    data = {
        "study_hours": [2, 3, 4, 5, 1, 6, 7, 3, 8, 2, 4, 5, 6, 7, 2, 3, 8, 9, 5, 4],
        "attendance": [60, 65, 70, 75, 55, 80, 85, 68, 90, 58, 72, 78, 82, 88, 62, 67, 91, 95, 77, 73],
        "assignments_completed": [3, 4, 5, 6, 2, 7, 8, 4, 9, 3, 5, 6, 7, 8, 3, 4, 9, 10, 6, 5],
        "previous_score": [45, 50, 55, 60, 40, 65, 70, 52, 78, 44, 58, 62, 68, 74, 47, 53, 80, 85, 64, 59],
        "internet_access": [
            "Yes", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "No", "Yes", "No",
            "Yes", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes"
        ],
        "final_score": [48, 54, 60, 66, 42, 72, 78, 56, 85, 46, 63, 68, 74, 81, 50, 57, 87, 92, 70, 64],
    }
    return pd.DataFrame(data)


def evaluate_model(name: str, y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Return common regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mse ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Build preprocessing pipeline for numeric and categorical columns."""
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def plot_predictions(y_true: pd.Series, y_pred: pd.Series, output_path: Path) -> None:
    """Save a scatter plot of actual vs predicted values."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_true, y=y_pred, s=80, color="teal")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="red", linestyle="--")
    plt.xlabel("Actual Final Score")
    plt.ylabel("Predicted Final Score")
    plt.title("Actual vs Predicted Student Scores")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> int:
    """Run the full ML workflow."""
    try:
        df = create_sample_dataset()

        features = ["study_hours", "attendance", "assignments_completed", "previous_score", "internet_access"]
        target = "final_score"

        X = df[features]
        y = df[target]

        numeric_features = ["study_hours", "attendance", "assignments_completed", "previous_score"]
        categorical_features = ["internet_access"]

        preprocessor = build_preprocessor(numeric_features, categorical_features)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ),
        }

        results: list[dict[str, float]] = []

        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            metrics = evaluate_model(model_name, y_test, predictions)
            results.append(metrics)

            print(f"\n{model_name}")
            print("-" * len(model_name))
            print(f"MAE  : {metrics['MAE']:.2f}")
            print(f"RMSE : {metrics['RMSE']:.2f}")
            print(f"R2   : {metrics['R2']:.2f}")

            if model_name == "Random Forest Regressor":
                plot_predictions(y_test, predictions, Path("actual_vs_predicted.png"))

        results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
        best_model = results_df.iloc[0]

        print("\nBest Model")
        print("----------")
        print(f"Model: {best_model['model']}")
        print(f"R2 Score: {best_model['R2']:.2f}")

        results_df.to_csv("model_results.csv", index=False)
        print("\nSaved:")
        print("- model_results.csv")
        print("- actual_vs_predicted.png")

        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())