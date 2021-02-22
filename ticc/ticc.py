from functools import wraps

import numpy as np
import pandas as pd

from joblib import delayed, Parallel
from sklearn import mixture
from sklearn.base import BaseEstimator, ClusterMixin

from .admm import ADMM


def pandas_handler(fn):
    @wraps(fn)
    def wrapper(self, X, *args, **kwargs):
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = pd.DataFrame(X)
            self.feature_names_ = X.columns
            X = X.to_numpy()
        elif not isinstance(X, np.ndarray):
            raise ValueError('Data type not supported')

        return fn(self, X, *args, **kwargs)
    return wrapper


class TICC(ClusterMixin, BaseEstimator):
    def __init__(
            self,
            window_size=10,
            n_clusters=5,
            lambda_parameter=11e-2,
            switch_penalty=400,
            max_iters=1000,
            threshold=2e-5,
            cluster_reassignment=20,
            biased=False,
            fill_labels=None,
            verbose=0,
            n_jobs=None,
    ):
        '''
        Refactoring of the TICC algorithm written by David Hallac and descirbed in the paper:
        https://stanford.edu/~boyd/papers/pdf/ticc.pdf

        Original repo:

        :param window_size: size of the sliding window
        :param n_clusters: number of clusters
        :param lambda_parameter: sparsity parameter
        :param switch_penalty: temporal consistency parameter
        :param max_iters: number of iterations
        :param threshold: convergence threshold
        :param cluster_reassignment: number of points to reassign to a 0 cluster
        :param biased: Using the biased or the unbiased covariance
        :param fill_labels: None or 'first', how to fill the initial labels that are lost due to the window_Size effect
        :param verbose: int, level of verbosity
        :param n_jobs: number of processe to spawn
        '''

        self.window_size = window_size
        self.n_clusters = n_clusters
        self.lambda_parameter = lambda_parameter
        self.switch_penalty = switch_penalty
        self.max_iters = max_iters
        self.threshold = threshold
        self.cluster_reassignment = cluster_reassignment
        self.num_blocks = self.window_size + 1
        self.biased = biased
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.fill_labels = fill_labels

    @pandas_handler
    def fit(self, X, y=None, **fit_params):
        X_stacked = self.stack_data(X)
        self.X_last_ = X[-self.window_size+1:] if self.window_size > 1 else None

        return self._fit(X_stacked)

    def _fit(self, X_stacked):
        # re-implemented
        '''
        Main fitting function which represents a variation of the EM algorithm
        Expects the X_stacked data as described in the paper of shape
        (n_samples - self.window_size + 1, n_features * self.window_size)

        :param X_stacked:
        :return: self
        '''
        assert self.max_iters > 0  # must have at least one iteration

        # Initialization
        # Gaussian Mixture
        gmm = mixture.GaussianMixture(n_components=self.n_clusters, covariance_type="full")
        gmm.fit(X_stacked)
        labels = gmm.predict(X_stacked)

        # PERFORM TRAINING ITERATIONS
        for i in range(self.max_iters):
            # Assuming gaussian mixture initialization always returns at least one point per cluster
            # 1. M-step runs the Toeplitz Graphical Lasso and updates the inverse covariance matrices for each cluster.
            thetas, means = self.m_step(X_stacked, labels=labels)

            # 2. E-step updates the cluster assignment keeping the inverse covariances fixed
            old_labels = labels.copy()
            labels, lle = self.e_step(X_stacked, thetas=thetas, means=means)

            # BREAK CONDITION
            if np.array_equal(old_labels, labels):
                break

            # 3. Reassign a few points to empty clusters
            labels = self.reassign_empty_clusters(labels, thetas)

            if i == self.max_iters - 1:
                # logic: if I hit the previous break condition it means that the e-step didn't modify the labels
                #        hence I don't need to recalculate means and clustered_indices as they are still valid
                clustered_indices = self.get_cluster_indices(labels)
                means = {cluster: X_stacked[cluster_indices].mean() for cluster, cluster_indices in clustered_indices}
                lle = self.smoothen_lle(X_stacked, thetas=thetas, means=means)

            if self.verbose:
                print(f'\nIteration {i}/{self.max_iters}')
                for cluster, cluster_indices in self.get_cluster_indices(labels).items():
                    print(f'\t AFTER: length of cluster {cluster} => {len(cluster_indices)}')

        # end of training
        # assigning first self.window_size - 1 observations to the first cluster
        self.labels_ = self.finalize_labels(labels)
        self.thetas_ = thetas
        self.means_ = means
        self.lle_ = lle
        return self

    @pandas_handler
    def predict(self, X):
        '''
        Assumes that test data starts right after the end of training data
        :param X: {array-like, sparse matrix} of shape (n_samples, n_features) New data to predict.
        :return labels: ndarray of shape (n_samples,). Index of the cluster each sample belongs to.
        '''
        X_test = np.concatenate([self.X_last_, X]) if self.window_size > 1 else X
        X_stacked = self.stack_data(X_test)

        lle = np.concatenate([self.lle_, self.smoothen_lle(X_stacked, thetas=self.thetas_, means=self.means_)])
        labels = self.update_clusters(lle, self.switch_penalty)
        return labels[-len(X):]

    def fit_predict(self, X, y=None):
        X_stacked = self.stack_data(X)
        self._fit(X_stacked)
        return self.labels_

    def stack_data(self, X):
        # re-implemented
        n_samples, n_features = X.shape
        X_stacked = np.concatenate([X[start:start + self.window_size].reshape(1, -1)
                                    for start in range(0, n_samples - self.window_size + 1)])

        return X_stacked

    def finalize_labels(self, labels):
        value = labels[0] if self.fill_labels else np.nan
        return np.concatenate((np.full(diff, value), labels)) if (diff := (self.window_size - 1)) else labels

    def get_cluster_indices(self, labels=None):
        labels = labels if labels is not None else self.labels_
        return {cluster: np.flatnonzero([labels == cluster]) for cluster in range(self.n_clusters)}

    def m_step(self, X_stacked, labels):
        '''
        The M-step corresponds to solving the Toeplitz Graphical Lasso which updates the inverse covariances
        :param X_stacked:
        :param clustered_indices:
        :return:
        '''
        n_features_stacked = X_stacked.shape[1]
        n_features = n_features_stacked // self.window_size

        solvers = {}
        means = {}
        for cluster, cluster_indices in self.get_cluster_indices(labels).items():
            x_cluster = X_stacked[cluster_indices]
            block_size = n_features
            prob_size = self.window_size * block_size
            lamb = np.zeros((prob_size, prob_size)) + self.lambda_parameter
            S = np.cov(x_cluster, bias=self.biased, rowvar=False)
            means[cluster] = x_cluster.mean(axis=0)

            rho = 1
            solvers.setdefault(cluster, ADMM(
                lamb=lamb,
                n_blocks=self.window_size,
                block_size=block_size,
                rho=rho,
                S=S
            ))

        res = Parallel(n_jobs=self.n_jobs)(
            delayed(solver.run)(
                max_iters=1000,
                eps_abs=1e-6,
                eps_rel=1e-6,
                verbose=False
            ) for solver in solvers.values())

        # this returns the updated inverse covariance matrices for each cluster
        return {cluster: solver.theta for cluster, solver in zip(solvers, res)}, means

    def e_step(self, X_stacked, thetas, means):
        # re-implemented
        '''
        The e-step calculates the lle for each sample i and each cluster j
        (probability that n-dimensional sample i belongs to cluster j) and uses this matrix to update the clusters
        based on a dynamic programming algorithm.

        Returns:
            vector of labels (one label for each sample)
        '''

        # SMOOTHENING
        smooth_lle = self.smoothen_lle(X_stacked, thetas=thetas, means=means)

        # Update cluster points - using NEW smoothening
        return self.update_clusters(smooth_lle, switch_penalty=self.switch_penalty), smooth_lle

    def smoothen_lle(self, X_stacked, thetas, means):
        # re-implemented
        '''
        This function calculates the lle given a fixed covariance matrices updated during the m-step
        :param X_stacked:
        :param labels:
        :param thetas: a dictionary with cluster as keys and inverse covariance matrices as values
        :return: lle matrix of shape len(X_stacked), self.n_cluster where each entry lle(i, j) is the probability that
                 the point i (which is n-dimensional according to n_features) belongs to cluster j
        '''

        n_samples, n_features_stacked = X_stacked.shape
        lle = np.zeros(shape=(n_samples, self.n_clusters))
        for cluster in range(self.n_clusters):
            inv_cov = thetas[cluster]
            cluster_mean = means[cluster]
            # if B = inv(A) then det(B) = 1 / det(A) => log(det(A)) = log(1 / det(B)) = - log(det(B))
            log_det_cov = - np.log(np.linalg.det(inv_cov))
            X_centered = X_stacked - cluster_mean
            lle[:, cluster] = ((X_centered @ inv_cov) * X_centered).sum(axis=1) + log_det_cov
        return lle

    @staticmethod
    def update_clusters(lle, switch_penalty=1):

        n_samples, n_clusters = lle.shape
        future_cost_vals = np.zeros(shape=(n_samples, n_clusters))

        # compute future costs
        for i in range(n_samples - 2, -1, -1):
            j = i + 1
            future_costs = future_cost_vals[j, :]
            lle_vals = lle[j, :]
            for cluster in range(n_clusters):
                total_vals = future_costs + lle_vals + switch_penalty
                total_vals[cluster] -= switch_penalty
                future_cost_vals[i, cluster] = np.min(total_vals)

        # compute the best path
        path = np.zeros(n_samples)

        # the first location
        curr_location = np.argmin(future_cost_vals[0, :] + lle[0, :])
        path[0] = curr_location

        # compute the path
        for i in range(n_samples - 1):
            j = i + 1
            future_costs = future_cost_vals[j, :]
            lle_vals = lle[j, :]
            total_vals = future_costs + lle_vals + switch_penalty
            total_vals[int(path[i])] -= switch_penalty

            path[i + 1] = np.argmin(total_vals)

        # return the computed path
        return path.astype(np.int32)

    def reassign_empty_clusters(self, labels, thetas):
        cluster_norms = [(np.linalg.norm(np.linalg.inv(theta)), cluster) for cluster, theta in thetas.items()]
        norms_sorted = sorted(cluster_norms, reverse=True)

        # clusters that are not 0 as sorted by norm
        all_cluster_indices = self.get_cluster_indices(labels)
        valid_clusters = [cluster for norm, cluster in norms_sorted if all_cluster_indices[cluster].size]
        empty_clusters = [c for c in all_cluster_indices if c not in valid_clusters]

        # Add a point to the empty clusters
        # Assuming more non empty clusters than empty ones
        counter = 0
        for cluster in empty_clusters:
            cluster_selected = valid_clusters[counter]  # a cluster that is not len 0
            counter = (counter + 1) % len(valid_clusters)

            if self.verbose:
                print(f"\tCluster {cluster} has 0 elements. Reassigning from cluster {cluster_selected}")

            # random point number from that cluster
            start_point = np.random.choice(all_cluster_indices[cluster_selected])
            for i in range(self.cluster_reassignment):
                # put cluster_reassignment points from point_num in this cluster
                point_to_move = start_point + i
                if point_to_move >= len(labels):
                    break
                labels[point_to_move] = cluster
        return labels
