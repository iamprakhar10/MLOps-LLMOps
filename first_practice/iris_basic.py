import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("iris-classification")

with mlflow.start_run():
    max_depth = 3
    model = RandomForestClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model,
                    artifact_path="model",
                    skops_trusted_types=["sklearn.tree._tree.Tree"],)