import os
import shutil
import time
import warnings
import logging

import mlflow
import mlflow.sklearn

import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
)

from sklearn.pipeline import (
    Pipeline,
)

from sklearn.preprocessing import (
    StandardScaler,
)

from sklearn.svm import (
    SVC,
)

from sklearn.neighbors import (
    KNeighborsClassifier,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from xgboost import (
    XGBClassifier,
)

sns.set_theme(style="whitegrid")
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")




class DietModelPipeline:
    def __init__(self, data_path='healthy_diet_calorie_intake_preprocessing.csv'):
        self.data_path = data_path
        self.target_names = ['Obese', 'Underweight', 'Overweight', 'Healthy']
        self.predictions = {}
    
        self.models = {

            "SVC":
            Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "clf",
                        SVC(
                            C=1.4330109455635685,
                            kernel="linear",
                            probability=True,
                            random_state=42,
                        ),
                    ),
                ]
            ),

            "KNN":
            Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "clf",
                        KNeighborsClassifier(
                            metric="manhattan",
                            n_neighbors=6,
                            weights="distance",
                        ),
                    ),
                ]
            ),

            "Gradient Boosting":
            GradientBoostingClassifier(
                learning_rate=0.14447746112718687,
                n_estimators=207,
                subsample=0.8542703315240834,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
            ),

            "Random Forest":
            RandomForestClassifier(
                bootstrap=True,
                class_weight="balanced_subsample",
                criterion="gini",
                max_depth=20,
                max_features="sqrt",
                min_samples_leaf=1,
                min_samples_split=2,
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),

            "XGBoost":
            XGBClassifier(
                colsample_bytree=0.9439761626945282,
                learning_rate=0.10113389297817889,
                max_depth=3,
                n_estimators=266,
                subsample=0.6053059844639466,
                objective="multi:softmax",
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            ),
        }

        mlflow.sklearn.autolog()
        # mlflow.set_tracking_uri("http://127.0.0.1:5001/") 
        mlflow.set_experiment("Diet_Health_Status_Basic")

    def prepare_data(
        self,
        leaked_columns=[
            "BMI",
            "Height_cm",
            "Weight_kg",
            "Health_Status",
        ],
    ):
        """
        Description:
            Loads dataset, removes target leakage columns,
            and splits data into training and testing sets.

        Args:
            leaked_columns (list):
                Columns that should be excluded
                from model training.

        Returns:
            None
        """

        df_model = pd.read_csv(
            self.data_path
        )

        columns_to_drop = [
            col
            for col in leaked_columns
            if col in df_model.columns
        ]

        X = df_model.drop(
            columns=columns_to_drop
        )

        y = df_model["Health_Status"]

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )


    def train_and_evaluate(
        self,
        report_path="model_base/classification_report.csv",
    ):
        """
        Description:
            Trains all models, evaluates performance,
            generates classification reports,
            ranks models, and saves the best model.

        Args:
            report_path (str):
                Path for aggregated report file.

        Returns:
            None
        """

        os.makedirs(
            "model_base",
            exist_ok=True,
        )

        print(
            "\n=== Model Training and Evaluation ===\n"
        )

        all_reports = []
        ranking_results = []

        best_model = None
        best_model_name = None
        best_f1 = 0

        for name, model in self.models.items():

            print(
                f"Training {name} ..."
            )

            start_time = time.time()

            with mlflow.start_run(
                run_name=f"Base_{name.replace(' ', '_')}",
                nested=True,
            ):

                model.fit(
                    self.X_train,
                    self.y_train,
                )

                y_pred = model.predict(
                    self.X_test
                )

                self.predictions[name] = y_pred

                accuracy = accuracy_score(
                    self.y_test,
                    y_pred,
                )

                precision = precision_score(
                    self.y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )

                recall = recall_score(
                    self.y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )

                f1 = f1_score(
                    self.y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )

                print(
                    f"Accuracy : {accuracy:.4f}"
                )

                print(
                    f"F1 Score : {f1:.4f}"
                )

                ranking_results.append(
                    {
                        "Model": name,
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1_Score": f1,
                    }
                )

                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model
                    best_model_name = name

                report_dict = classification_report(
                    self.y_test,
                    y_pred,
                    target_names=self.target_names,
                    output_dict=True,
                    zero_division=0,
                )

                df_report = pd.DataFrame(
                    report_dict
                ).transpose()

                df_report["Model"] = name
                df_report["Metric_Class"] = df_report.index

                all_reports.append(
                    df_report
                )

            elapsed = (
                time.time()
                - start_time
            )

            mins, secs = divmod(
                elapsed,
                60,
            )

            print(
                f"Completed in "
                f"{int(mins)}m {int(secs)}s\n"
            )

        final_report = pd.concat(
            all_reports,
            ignore_index=True,
        )

        final_report.to_csv(
            report_path,
            index=False,
        )

        ranking_df = pd.DataFrame(
            ranking_results
        )

        ranking_df = ranking_df.sort_values(
            by="F1_Score",
            ascending=False,
        )

        ranking_df.to_csv(
            "model_base/model_ranking.csv",
            index=False,
        )

        print("\n=== Model Ranking ===")
        print(
            ranking_df.to_string(
                index=False
            )
        )

        print(
            f"\nBest Model : "
            f"{best_model_name}"
        )

        print(
            f"Best F1 Score : "
            f"{best_f1:.4f}"
        )

        if os.path.exists(
            "model_base/saved_model"
        ):
            shutil.rmtree(
                "model_base/saved_model"
            )

        custom_env = {
            "name": "diet-env",
            "channels": [
                "conda-forge"
            ],
            "dependencies": [
                "python=3.12.7",
                "pip",
                {
                    "pip": [
                        "mlflow",
                        "scikit-learn",
                        "pandas",
                        "matplotlib",
                        "seaborn",
                        "xgboost",
                    ]
                },
            ],
        }

        mlflow.sklearn.save_model(
            best_model,
            "model_base/saved_model",
            conda_env=custom_env,
        )

        print(
            "\nBest model successfully saved."
        )

    def plot_confusion_matrix(
        self,
        save_path="model_base/confusion_matrix_all_models.png",
    ):
        """
        Description:
            Generates confusion matrix visualizations
            for all trained models and saves them
            into a single image.

        Args:
            save_path (str):
                Output image path.

        Returns:
            None
        """

        os.makedirs(
            "model_base",
            exist_ok=True,
        )

        n_models = len(
            self.predictions
        )

        fig, axes = plt.subplots(
            1,
            n_models,
            figsize=(6 * n_models, 5),
        )

        if n_models == 1:
            axes = [axes]

        for ax, (name, y_pred) in zip(
            axes,
            self.predictions.items(),
        ):

            cm = confusion_matrix(
                self.y_test,
                y_pred,
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.target_names,
                yticklabels=self.target_names,
                ax=ax,
            )

            ax.set_title(
                name,
                fontsize=12,
                fontweight="bold",
            )

            ax.set_xlabel(
                "Predicted Label"
            )

            ax.set_ylabel(
                "Actual Label"
            )

        plt.tight_layout()

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"Confusion matrix saved to "
            f"'{save_path}'"
        )


if __name__ == "__main__":

    pipeline = DietModelPipeline(
        data_path=(
            "healthy_diet_calorie_intake_preprocessing.csv"
        )
    )

    pipeline.prepare_data()

    pipeline.train_and_evaluate(
        report_path=(
            "model_base/classification_report.csv"
        )
    )

    pipeline.plot_confusion_matrix(
        save_path=(
            "model_base/confusion_matrix_all_models.png"
        )
    )

    print(
        "\nTraining pipeline completed successfully."
    )

