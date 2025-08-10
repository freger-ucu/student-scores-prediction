from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import BaseEstimator


def evaluate(models: dict[str, BaseEstimator], k, X, y) -> tuple[dict, dict]:
    """
    Evaluates the performance of multiple models using cross-validation.

    :param models: dict, dictionary of model names and their instances
    :param k: int, number of folds for cross-validation
    :param X: pd.DataFrame, feature matrix
    :param y: pd.Series, target variable
    :return: tuple of two dicts, first with MAE scores and second with R2 scores
    """
    mae_scores = {name: [] for name in models}
    r2_scores = {name: [] for name in models}

    for name, model in models.items():
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mae_scores[name].append(mean_absolute_error(y_test, preds))
            r2_scores[name].append(r2_score(y_test, preds))

    return mae_scores, r2_scores


def test_hyperparameter(params, to_tune, tune_grid, model, X, y):
    """
    Tests a hyperparameter by training the model with different values and evaluating performance.

    :param params: dict, fixed parameters for the model
    :param to_tune: str, the name of the parameter to tune
    :param tune_grid: list, values to test for the parameter
    :param model: BaseEstimator, the model instance to train and evaluate
    :param X: pd.DataFrame, feature matrix for training and testing
    :param y: pd.Series, target variable for training and testing
    :return: list of MAE scores for each tuned value
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)

    results = []

    for tune in tune_grid:
        params[to_tune] = tune
        model_instance = model(**params)
        model_instance.fit(X_train, y_train)
        preds = model_instance.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        results.append(mae)
        print(f"Testing {to_tune}={tune}: MAE={mae:.4f}")

    return results


def plot_hyperparameter(tune_grid, results, to_tune, title, x_log=False, y_log=False):
    """
    Plots the results of hyperparameter tuning.

    :param tune_grid: list, values tested for the hyperparameter
    :param results: list, MAE scores corresponding to each value in tune_grid
    :param to_tune: str, the name of the hyperparameter being tuned
    :param x_log: bool, whether to use logarithmic scale for x-axis
    :param y_log: bool, whether to use logarithmic scale for y-axis
    :param title: str, title for the plot
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(tune_grid, results, marker='o')
    plt.title(title)
    plt.xlabel(to_tune)
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.grid()
    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')
    plt.show()
