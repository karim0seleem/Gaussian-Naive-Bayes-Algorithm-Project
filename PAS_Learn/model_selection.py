import numpy as np

def My_train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split the dataset into train and test subsets.

    Parameters:
    X : DataFrame, shape (n_samples, n_features)
        The input data.
    y : Series, shape (n_samples,)
        The target labels.
    test_size : float, int, None, optional (default=0.25)
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.
    random_state : int or None, optional (default=None)
        Random state for shuffling and splitting.

    Returns:
    X_train : DataFrame, shape (n_train_samples, n_features)
        The training input data.
    X_test : DataFrame, shape (n_test_samples, n_features)
        The testing input data.
    y_train : Series, shape (n_train_samples,)
        The training target labels.
    y_test : Series, shape (n_test_samples,)
        The testing target labels.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate random indices
    indices = np.random.permutation(len(X))

    if isinstance(test_size, float):
        test_size = int(len(X) * test_size)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    # Split X and y based on indices
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    return X_train, X_test, y_train, y_test
