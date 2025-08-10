"""
Module containing custom implementations of
Linear Regression (using mini-batch gradient descent)
and KNN Regression models.
"""
import numpy as np
import pandas as pd


class LinearRegression:
    """
    Linear regression using mini-batch gradient descent.
    """

    def __init__(self, learning_rate=0.00005, epochs=100, batch_size=32, debug=False, print_every=100):
        """
        Initialize a model
        :param X: pd.DataFrame of training features (no intercept column)
        :param Y: pd.Series or array of training targets
        :param learning_rate: float
        :param epochs: int, number of full passes over the data
        :param batch_size: int, size of each mini-batch
        :param debug: bool, enable/disable RSS debug every (print_every) epochs
        :param print_every: int, how often (in epochs) to print RSS
        """
        # Add intercept column
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.print_every = print_every
        self.coeffs = np.array([0])
        self.debug = debug

    def rss(self, X, Y):
        """
        Get residual sum of squares for multiple features

        :return: Residual sum of squares (float)
        """
        preds = X.dot(self.coeffs)
        return np.sum((Y - preds) ** 2)

    def fit(self, X, Y):
        """
        Trains a model to minimize the RSS, tuning the coefficients for the train data, using Mini-batch gradient descent
        :return: None
        """
        X = pd.concat(
            [pd.DataFrame(np.ones((X.shape[0], 1)), columns=['Intercept']),
             X.reset_index(drop=True)],
            axis=1
        ).to_numpy()

        Y = Y.reset_index(drop=True).to_numpy()

        self.coeffs = np.zeros(X.shape[1])

        n_samples = X.shape[0]

        for epoch in range(1, self.epochs + 1):
            perm = np.random.permutation(n_samples)
            X_shuf = X[perm]
            y_shuf = Y[perm]

            # iterate mini-batches
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]

                preds = X_batch.dot(self.coeffs)
                errors = preds - y_batch

                grad = 2 * X_batch.T.dot(errors) / X_batch.shape[0]

                self.coeffs -= self.learning_rate * grad

            if self.debug and (epoch % self.print_every == 0 or epoch == 1):
                print(f"Epoch {epoch:4d} — RSS: {self.rss(X, Y):.2f}")

    def predict(self, X_test):
        """
        Predicts the target values for all rows
        :param X_test: features of all test cases
        :return: pd Dataframe of target variables for each test case
        """
        return X_test.dot(np.array(self.coeffs[1:])) + np.array(self.coeffs[0])


class KNNRegression:
    def __init__(self, k=3):
        """
        Initialize the KNN Regressor
        :param k: the number of nearest neighbors to consider
        """
        self.k = k

    def fit(self, X, Y):
        """
        Store the training data.
        :param X: Training features
        :param Y: Training target values
        """
        self.X_train = X
        self.Y_train = np.array(Y)

    def predict(self, X_test):
        """
        Predict the target values for the test data.
        :param X_test: Test features
        :return: Predicted target values
        """
        X_test = np.array(X_test)
        predictions = []

        for point in X_test:
            # Compute distances from the test point to all training points
            per_axis_distances = self.X_train - point
            distances = np.linalg.norm(per_axis_distances, axis=1)

            # Get indices of the k nearest neighbors
            sorted_indices = np.argsort(distances)
            k_indices = sorted_indices[:self.k]

            # Compute the average of the target values of the k nearest neighbors
            k_nearest_targets = self.Y_train[k_indices]
            predictions.append(np.mean(k_nearest_targets))

        return np.array(predictions)
