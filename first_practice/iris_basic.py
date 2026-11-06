import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_experiment("iris-classification")

with mlflow.start_run():
    n_estimators = 100
    max_depth = 3
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_params({
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": 42,
        "test_size": 0.2,
        "stratify": True,
    })
    mlflow.log_metric("accuracy", accuracy)

    # Save the prediction plot as an artifact in this run.
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=model.classes_,
        display_labels=iris.target_names,
        cmap="Blues",
    )
    display.ax_.set_title("Iris test predictions")
    display.figure_.tight_layout()
    mlflow.log_figure(display.figure_, "plots/confusion_matrix.png")
    plt.close(display.figure_)

    # Describe the model's expected inputs and prediction outputs.
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        input_example=X_train[:5],
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
