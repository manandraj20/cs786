import numpy as np
from random import shuffle
from scipy.stats import multivariate_normal

class dLocalMAP:
    """
    Modified Rational Model of Categorization for continuous data.
    """
    def __init__(self, args):
        self.partition = [[]]  # Initialize clusters
        self.c, self.alpha = args  # c: new cluster probability, alpha: prior
        self.alpha0 = sum(self.alpha.T)
        self.N = 0  # Total number of data points seen
        self.cluster_means = []  # Store means of clusters
        self.cluster_covs = []  # Store covariances of clusters

    def gaussian_prob(self, stim, k):
        """
        Calculate the probability of the stimulus belonging to cluster k
        using a Gaussian distribution.
        """
        if len(self.partition[k]) == 0:
            return 1  # For a new cluster, assign equal probability
        mean = self.cluster_means[k]
        cov = self.cluster_covs[k]
        return multivariate_normal.pdf(stim, mean=mean, cov=cov)

    def condclusterprob(self, stim, k):
        """
        Calculate P(F|k) using Gaussian probability.
        """
        return self.gaussian_prob(stim, k)

    def posterior(self, stim):
        """
        Calculate P(k|F) for each cluster.
        """
        pk = np.zeros(len(self.partition))
        pFk = np.zeros(len(self.partition))

        for k in range(len(self.partition)):
            pk[k] = self.c * len(self.partition[k]) / ((1 - self.c) + self.c * self.N)
            if len(self.partition[k]) == 0:  # New cluster
                pk[k] = (1 - self.c) / ((1 - self.c) + self.c * self.N)
            pFk[k] = self.condclusterprob(stim, k)

        pkF = pk * pFk  # Unnormalized posterior
        return pkF

    def stimulate(self, stim):
        """
        Assign the stimulus to the cluster with the highest posterior probability.
        """
        winner = np.argmax(self.posterior(stim))

        if len(self.partition[winner]) == 0:
            self.partition.append([])
            self.cluster_means.append(np.array(stim))  # Initialize mean
            self.cluster_covs.append(np.eye(len(stim)))  # Initialize covariance
        else:
            # Update cluster mean and covariance
            self.cluster_means[winner] = (
                self.cluster_means[winner] * len(self.partition[winner]) + np.array(stim)
            ) / (len(self.partition[winner]) + 1)
            self.cluster_covs[winner] = np.cov(
                np.array(self.partition[winner] + [stim]).T
            )

        self.partition[winner].append(stim)
        self.N += 1

    def query(self, stimulus):
        """
        Predict the category for a new data point.
        """
        pkF = self.posterior(stimulus)
        pkF = pkF[:-1] / (sum(pkF[:-1]) + 1e-10)  # Exclude new cluster probability

        # Calculate probability of belonging to each category
        pjF = np.array(
            [
                sum(
                    [
                        pkF[k] * self.gaussian_prob(stimulus, k)
                        for k in range(len(self.partition) - 1)
                    ]
                )
                for j in range(len(self.alpha[0]))  # Assuming alpha is the same for all dimensions
            ]
        )

        return pjF / (sum(pjF)+1e-10)


def load_and_normalize_data(X_file, y_file):
    """
    Load data from CSV files and normalize.
    """
    X = np.loadtxt(X_file, delimiter=",")
    y = np.loadtxt(y_file, delimiter=",")

    # Normalize data (important for Gaussian probabilities)
    X[:, 0] = (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min())
    X[:, 1] = (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min())
    y[:, 0] = (y[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min())
    y[:, 1] = (y[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min())

    return X, y


def main():
    X_file = "X.csv"
    y_file = "y.csv"
    X, y = load_and_normalize_data(X_file, y_file)

    # Model parameters
    c = 0.5  # Probability of new cluster
    alpha = np.ones((2, 3))  # Prior (adjust as needed)

    model = dLocalMAP([c, alpha])

    # Train the model
    for data_point in X:
        model.stimulate(data_point[:2])  # Use only height and weight

    # Predict categories for new data
    predictions = []
    for data_point in y:
        pred = model.query(data_point)
        predicted_category = np.argmax(pred) + 1  # Add 1 to match your category labels
        predictions.append(predicted_category)

    print("Predictions:", predictions)


if __name__ == "__main__":
    main()