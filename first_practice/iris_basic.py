from collections import Counter
from pathlib import Path
import subprocess
from time import perf_counter

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

PARAM_GRID = {"n_estimators": [100, 200], "max_depth": [3, None]}

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def log_git_details():

    """Record the commit and edits, since a commit misses uncommitted code."""

    def git(*args):
        """Run Git in the project directory and return its text output."""


        """
        Typical return: stripped Git output as a string, such as "e930201..."
        for rev-parse HEAD (hash shortened here), or "" for a clean status.
        """

        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), *args],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()

    try:
        commit = git("rev-parse", "HEAD")

        """Porcelain is Git's stable status format; the HEAD diff captures tracked-file edits."""

        status = git("status", "--porcelain")

        diff = git("diff", "HEAD", "--", ".")
    except (OSError, subprocess.CalledProcessError):

        """
        Tags are searchable run metadata. Record unavailable Git information without
        stopping training; this tag belongs to the currently active parent run.
        """

        mlflow.set_tag("git.available", "false")
    else:

        """
        set_tags accepts a dictionary of metadata labels. mlflow.source.git.commit
        links the run to HEAD; git.dirty records whether local changes also exist.
        A commit hash alone does not capture uncommitted edits.
        """

        mlflow.set_tags({
            "git.available": "true",
            "mlflow.source.git.commit": commit,

            "git.dirty": str(bool(status)).lower(),
        })

        """
        log_text(text, artifact_file) saves text as a run artifact. The second
        argument is a path inside the run artifact store, not a required local file.
        """

        mlflow.log_text(status or "Clean working tree", "source/git_status.txt")
        if diff:

            """
            Save tracked-file changes relative to HEAD alongside the commit hash.
            The patch does not include the contents of untracked files.
            """

            mlflow.log_text(diff, "source/uncommitted_changes.patch")

    """
    log_artifact(local_path, artifact_path=...) copies an existing file into
    the run artifact store. Here it saves the actual training script under
    source/, including edits that have not yet been committed.
    """

    mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="source")

    """
    Typical return: None; Git details and source files are logged to MLflow.
    """


def classification_metrics(y_true, predictions, prefix):
    """Return accuracy and macro-averaged scores with prefixed metric names."""

    """Macro averaging gives each class equal weight; zero_division=0 handles undefined ratios."""


    """
    Typical return with prefix="test" (illustrative scores):
    {"test_accuracy": 0.90, "test_f1_macro": 0.89,
     "test_precision_macro": 0.90, "test_recall_macro": 0.89}
    """

    return {
        f"{prefix}_accuracy": accuracy_score(y_true, predictions),
        f"{prefix}_f1_macro": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
        f"{prefix}_precision_macro": precision_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
        f"{prefix}_recall_macro": recall_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }

def main():
    """Compare Iris models, evaluate the winner, and log results to MLflow."""

    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=iris.target,
    )
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}

    """
    Select the experiment that groups these runs; create it if needed.
    This selects an experiment, not a database. Tracking storage still comes
    from your configured tracking URI or the default working-directory location.
    """

    mlflow.set_experiment("iris-classification")

    """
    Create a parent run for this search. run_name is a readable label and may
    repeat; MLflow assigns a separate unique run ID on each execution.
    The context manager finishes the run on normal exit and marks an unhandled
    failure accordingly. Logging calls below target the active run.
    """

    with mlflow.start_run(run_name="random-forest-hyperparameter-search"):

        """
        Describe the run with searchable tags. These labels do not configure the
        Random Forest. mlflow.note.content supplies the run description in the UI.
        """

        mlflow.set_tags({
            "model_type": "random_forest",
            "dataset": "iris",
            "selection_metric": "cv_f1_macro_mean",
            "mlflow.note.content": (
                "Compare four configurations with stratified CV on training data. "
                "Evaluate only the winner on the held-out test set."
            ),
        })
        log_git_details()

        """
        log_params accepts a dictionary of configuration values and records them
        in the parent run Parameters section. Parameters describe settings;
        metrics record measured numeric results. Parameter values are stored as text.
        """

        mlflow.log_params({
            "test_size": TEST_SIZE,
            "stratify": True,
            "cv_folds": CV_FOLDS,
            "cv_shuffle": True,
            "cv_random_state": RANDOM_STATE,
        })

        """
        log_dict(dictionary, artifact_file) serializes a dictionary as an artifact.
        PARAM_GRID is the complete search space: two tree counts and two depths.
        search/parameter_grid.json is our chosen path inside this run artifact store.
        The .json extension selects JSON output; Python None becomes JSON null.
        MLflow creates the artifact; no local search directory or file is required.
        This preserves what was searched; it does not execute the search.
        """

        mlflow.log_dict(PARAM_GRID, "search/parameter_grid.json")

        for name, features, targets in (
            ("training", X_train, y_train),
            ("testing", X_test, y_test),
        ):

            """
            from_numpy(features, targets=..., name=...) builds an MLflow dataset
            description with schema and a content digest. It does not save a full
            snapshot of these arrays. name is a readable dataset label.
            """

            dataset = mlflow.data.from_numpy(
                features,
                targets=targets,
                name=f"iris-{name}",
            )

            """
            Associate the dataset description with the active run. context labels
            its role here as training or testing; it does not split or evaluate data.
            """

            mlflow.log_input(dataset, context=name)

        """
        Save dataset provenance and class counts as data/dataset_info.json.
        The dictionary is the file content; the second argument below chooses
        its run-relative artifact path. This metadata is separate from model parameters.
        """

        mlflow.log_dict({
            "source": "sklearn.datasets.load_iris",
            "scikit_learn_version": sklearn.__version__,
            "feature_names": list(iris.feature_names),
            "class_names": iris.target_names.tolist(),
            "total_rows": len(iris.target),
            "train_rows": len(y_train),
            "test_rows": len(y_test),
            "train_class_counts": {
                iris.target_names[int(label)]: int(count)
                for label, count in Counter(y_train).items()
            },
            "test_class_counts": {
                iris.target_names[int(label)]: int(count)
                for label, count in Counter(y_test).items()
            },
        }, "data/dataset_info.json")

        best_score = float("-inf")
        best_params = None
        best_run_id = None
        search_results = []
        search_start = perf_counter()
        for params in ParameterGrid(PARAM_GRID):

            """
            nested=True creates a child run linked to the active parent search run.
            Each child holds one candidate configuration and its CV results.
            candidate exposes its run ID. Child runs do not automatically inherit
            parent tags or parameters; the parent becomes active again on child exit.
            """

            with mlflow.start_run(
                run_name=f"trees-{params['n_estimators']}-depth-{params['max_depth']}",
                nested=True,
            ) as candidate:

                model = RandomForestClassifier(**params, random_state=RANDOM_STATE)

                """
                Record every estimator parameter, including defaults, on this child run.
                This captures more configuration than logging only the two searched values.
                """

                mlflow.log_params(model.get_params())
                scores = cross_validate(
                    model,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=True,
                    error_score="raise",
                )
                metrics = {}

                """Cross-validation's test scores are validation folds, not the held-out test set."""

                for metric in scoring:

                    for split, prefix in (("train", "cv_train"), ("test", "cv")):
                        values = scores[f"{split}_{metric}"]

                        metrics[f"{prefix}_{metric}_mean"] = float(values.mean())

                        metrics[f"{prefix}_{metric}_std"] = float(values.std())

                metrics["cv_fit_seconds_mean"] = float(scores["fit_time"].mean())

                """
                log_metrics accepts a dictionary of numeric results. Here it saves CV
                training/validation means, standard deviations, and mean fit duration.
                cv_* refers to validation folds within X_train, not the held-out test set.
                """

                mlflow.log_metrics(metrics)

                """
                Preserve individual fold scores and timings as evaluation/cv_fold_scores.json
                in this child run. Converting NumPy arrays to lists makes them JSON-serializable.
                """

                mlflow.log_dict(
                    {key: values.tolist() for key, values in scores.items()},
                    "evaluation/cv_fold_scores.json",
                )
                search_results.append({
                    "run_id": candidate.info.run_id,
                    "params": params,
                    **metrics,
                })

                if metrics["cv_f1_macro_mean"] > best_score:
                    best_score = metrics["cv_f1_macro_mean"]
                    best_params = params
                    best_run_id = candidate.info.run_id

        """
        After the child runs close, these calls log to the parent again.
        log_metric(name, value) records one numeric result; search_seconds includes
        cross-validation and candidate logging time.
        """

        mlflow.log_metric("search_seconds", perf_counter() - search_start)

        """
        Save all candidate summaries in one parent artifact. The dictionary wraps
        the results list under candidates; search/results.json is its artifact path.
        """

        mlflow.log_dict({"candidates": search_results}, "search/results.json")

        """
        Store the winning child run ID as a tag so its detailed results can be found.
        This is a reference label, not model registration or a deployment action.
        """

        mlflow.set_tag("best_candidate_run_id", best_run_id)

        """
        Expose the winning validation macro F1 directly in the parent Metrics section.
        """

        mlflow.log_metric("best_cv_f1_macro_mean", best_score)

        model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE)

        """
        Log the selected model configuration on the parent, where the final model
        will be saved. Candidate parameters remain in their separate child runs.
        """

        mlflow.log_params(model.get_params())
        fit_start = perf_counter()
        model.fit(X_train, y_train)

        """
        Record final fitting duration in seconds as a parent-run metric.
        """

        mlflow.log_metric("training_seconds", perf_counter() - fit_start)

        """
        Log training accuracy, macro F1, precision, and recall under train_* keys.
        These help compare fit quality with held-out performance.
        """

        mlflow.log_metrics(classification_metrics(y_train, model.predict(X_train), "train"))
        predict_start = perf_counter()
        predictions = model.predict(X_test)

        """
        Record elapsed time for this batch of 30 predictions. This is a local
        timing measurement, not a production serving benchmark.
        """

        mlflow.log_metric("test_prediction_seconds", perf_counter() - predict_start)

        """
        Log final held-out accuracy, macro F1, precision, and recall under test_*
        keys. These results are measured after selection and do not choose the winner.
        """

        mlflow.log_metrics(classification_metrics(y_test, predictions, "test"))

        """
        Save the detailed per-species report as an artifact instead of flattening
        it into metric names. output_dict=True produces the dictionary log_dict
        requires; evaluation/test_classification_report.json below is its destination.
        """

        mlflow.log_dict(
            classification_report(
                y_test,
                predictions,
                labels=model.classes_,
                target_names=iris.target_names,
                output_dict=True,
                zero_division=0,
            ),
            "evaluation/test_classification_report.json",
        )

        display = ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            labels=model.classes_,
            display_labels=iris.target_names,
            cmap="Blues",
        )
        display.ax_.set_title("Selected model: Iris test predictions")
        display.figure_.tight_layout()

        """
        log_figure(figure, artifact_file) renders and saves the Matplotlib figure
        as plots/confusion_matrix.png in the parent run. No local PNG is required.
        The file is viewable in the UI under the run Artifacts section.
        """

        mlflow.log_figure(display.figure_, "plots/confusion_matrix.png")
        plt.close(display.figure_)

        """
        Save the fitted winner as a logged MLflow model named model. This packages
        the learned estimator with model metadata and dependency/environment files.

        signature: infer_signature(inputs, outputs) describes the expected batch
        shape and types: four floating-point features and integer predictions.
        MLflow pyfunc/serving uses this contract for input validation. NumPy inputs
        do not provide feature names; a signature cannot detect every ordering mistake.

        input_example: save five actual input rows as an example request.
        skops_trusted_types: allow the listed tree type when loading skops models;
        this is a serialization setting, not a hyperparameter.

        Only this final model is saved; child runs contain comparison results.
        Logging a model does not register it in the Model Registry.
        """

        mlflow.sklearn.log_model(
            model,
            name="model",

            signature=infer_signature(X_train, model.predict(X_train)),
            input_example=X_train[:5],

            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        print(f"Selected parameters: {best_params}")
        print(f"Best CV macro F1: {best_score:.4f}")
        print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")

    """
    Typical return: None; models/results are logged to MLflow and a summary is printed.
    """


if __name__ == "__main__":
    main()
