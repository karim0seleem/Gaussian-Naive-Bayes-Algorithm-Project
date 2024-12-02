import numpy as np
import pandas as pd


import numpy as np


class GaussianNaiveBayes:
    def __init__(self):

        self.class_priors = {}
        self.class_means = {}
        self.class_variances = {}

    def fit(self, X_train, y_train):
        """
        Fits the Gaussian Naive Bayes model to the training data.

        Parameters:
            X_train (ndarray): The training input samples of shape (n_samples, n_features).
            y_train (ndarray): The target values of shape (n_samples,).

        Returns:
            None
        """
        n_samples, n_features = X_train.shape  # number of samples and features
        classes = np.unique(y_train)  # unique classes aka our label values

        # Calculate class priors
        for c in classes:
            self.class_priors[c] = np.sum(y_train == c) / n_samples  # P(Y)

        # Calculate class means and variances
        for c in classes:
            X_c = X_train[y_train == c]  # all samples of class c
            self.class_means[c] = X_c.mean(axis=0)  # mean
            self.class_variances[c] = X_c.var(axis=0)  # variance also known as (standard deviation^2)

    def _gaussian_pdf(self, x, mean, variance):
        """
        Calculate the probability density function value of a Gaussian distribution given the input parameters.

        Parameters:
            x (float): The input value.
            mean (float): The mean of the distribution.
            variance (float): The variance of the distribution.

        Returns:
            float: The probability density function value.
        """
        exponent = -0.5 * ((x - mean) ** 2) / (variance)
        return np.exp(exponent) / np.sqrt(2 * np.pi * variance)

    def predict(self, X_test):
        """
        Predicts the class labels for the given test data.

        Args:
            X_test (pandas.DataFrame): The test data with features to predict the class labels for.

        Returns:
            list: A list of predicted class labels for each row in X_test.
        """
        predictions = []
        for _, x in X_test.iterrows():  # for each row in X_test
            posteriors = []  # store posteriors for each class
            for c in self.class_priors:  # for each class
                prior, mean, variance = self.class_priors[c], self.class_means[c], self.class_variances[c]  # P(Y), mean, variance

                likelihood = np.prod(self._gaussian_pdf(x, mean, variance))  # P(X|H)
                posterior = prior * likelihood  # P(Y) * P(Xi|Y)
                posteriors.append((c, posterior))  # store posterior for each class
            predicted_class = max(posteriors, key=lambda x: x[1])[0]  # get class with highest posterior
            predictions.append(predicted_class)  # store predicted class
        return predictions

'''
posteriors = [('A', 0.8), ('B', 0.6), ('C', 0.9)]

        posteriors           <- list of tuples
           |
           V
   +---------------+
   |  ('A', 0.8)   |
   +---------------+
   |  ('B', 0.6)   |
   +---------------+
   |  ('C', 0.9)   |   <- max() finds this tuple
   +---------------+
           |
           V
max(posteriors, key=lambda x: x[1])
           |
           V
       ('C', 0.9)
           |
           V
     ('C' is assigned to predicted_class)

'''