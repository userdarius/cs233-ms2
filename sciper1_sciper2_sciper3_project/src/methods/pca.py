import numpy as np

## MS2

class PCA(object):
    """
    PCA dimensionality reduction class.     Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.    """

    def __init__(self, d: int):
        """
        Initialize the new object (see dummy_methods.py)        and set its arguments.
        Arguments:            d (int): dimensionality of the reduced space        """
        self.d = d

        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None

    @property
    def pca(self) -> np.ndarray:
        if self.W is None:
            raise ValueError("PCA not computed")
        return self.W

    def find_principal_components(self, training_data) -> float:
        """
        Finds the principal components of the training data and returns the explained variance in percentage.
        IMPORTANT:            This function should save the mean of the training data and the kept principal components as
        self.mean and self.W, respectively.

        Arguments:            training_data (array): training data of shape (N,D)
        Returns:            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])        """
        training_data: np.ndarray = np.array(training_data)

        self.mean = np.mean(training_data, axis=0)
        centered_data: np.ndarray = training_data - self.mean
        cov: np.ndarray = np.cov(centered_data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        sorted_indices: np.ndarray = np.argsort(eigenvalues)[::-1]
        eigenvalues: np.ndarray = eigenvalues[sorted_indices]
        eigenvectors: np.ndarray = eigenvectors[:, sorted_indices]

        self.W = eigenvectors[:, :self.d]
        total_var: float = np.sum(eigenvalues)
        explained_var: float = np.sum(eigenvalues[:self.d])
        exvar: float = explained_var / total_var * 100
        return exvar

    def reduce_dimension(self, data) -> np.ndarray:
        """
        Reduce the dimensionality of the data using the previously computed components.
        Arguments:            data (array): data of shape (N,D)
        Returns:            data_reduced (array): reduced data of shape (N,d)
        """
        if self.mean is None or self.W is None:
            raise ValueError("PCA not computed")

        data: np.ndarray = np.array(data)

        centered_data: np.ndarray = data - self.mean
        data_reduced: np.ndarray = np.dot(centered_data, self.W)

        return data_reduced
        

