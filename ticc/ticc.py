from collections import defaultdict

import numpy as np
import pandas as pd

from joblib import delayed, Parallel
from sklearn import mixture
from sklearn.base import BaseEstimator

from .utils import (
    upper_to_full,
)
from .admm import ADMM


class TICC(BaseEstimator):
    def __init__(
            self,
            window_size=10,
            n_clusters=5,
            lambda_parameter=11e-2,
            switch_penalty=400,
            max_iters=1000,
            threshold=2e-5,
            n_jobs=1,
            compute_bic=False,
            cluster_reassignment=20,
            biased=False
    ):
        """
        Parameters:
            - window_size: size of the sliding window
            - n_clusters: number of clusters
            - lambda_parameter: sparsity parameter
            - switch_penalty: temporal consistency parameter
            - max_iters: number of iterations
            - threshold: convergence threshold
            - cluster_reassignment: number of points to reassign to a 0 cluster
            - biased: Using the biased or the unbiased covariance
        """
        self.window_size = window_size
        self.n_clusters = n_clusters
        self.lambda_parameter = lambda_parameter
        self.switch_penalty = switch_penalty
        self.max_iters = max_iters
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.compute_bic = compute_bic
        self.cluster_reassignment = cluster_reassignment
        self.num_blocks = self.window_size + 1
        self.biased = biased

        pd.set_option('display.max_columns', 500)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
        np.random.seed(102)

    def fit(self, X, y=None):
        """
        Main method for TICC solver.
        Parameters:
            - input_file: location of the data file
        """
        assert self.max_iters > 0  # must have at least one iteration
        self.log_parameters()

        # Stack the training data
        X_stacked = self.stack_data(X)

        # Initialization
        # Gaussian Mixture
        gmm = mixture.GaussianMixture(n_components=self.n_clusters, covariance_type="full")
        gmm.fit(X_stacked)

        history = dict(
            clustered_points=gmm.predict(X_stacked),
            old_clustered_points=None,  # points from last iteration
            clustered_points_group=None,
            cluster_mean={},
            cluster_mean_stacked={},
            opt_res=None,
            computed_covariance={},
            old_computed_covariance={},
            empirical_covariances={},
            train_cluster_inverse={},
            log_det_values={},  # log dets of the thetas
        )

        # PERFORM TRAINING ITERATIONS
        for i in range(self.max_iters):

            print("\n\n\nITERATION ###", i)
            history['clustered_points_group'] = self.get_clustered_points_group(history['clustered_points'])

            # train_clusters holds the indices in X_stacked
            # for each of the clusters
            self.train_clusters(X_stacked, history)

            self.optimize_clusters(history)
            print("UPDATED THE OLD COVARIANCE")

            # history['clustered_points'] will be overwritten after I call self.update_clusters
            history['old_clustered_points'] = history['clustered_points']
            self.update_clusters(X_stacked, history)

            if i != 0:
                # here I modify history['clustered_points']
                self.reassign_empty_clusters(X_stacked, history)

            print('\n')
            for cluster, cluster_indices in history['clustered_points_group'].items():
                print(f'AFTER UPDATE length of the cluster {cluster} ------> {len(cluster_indices)}')

            if np.array_equal(history['old_clustered_points'], history['clustered_points']):
                print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                break

        # end of training
        self.history_ = history
        return self

    def predict(self, X):
        '''

        :param X: {array-like, sparse matrix} of shape (n_samples, n_features) New data to predict.
        :return labels: ndarray of shape (n_samples,). Index of the cluster each sample belongs to.
        '''

        X_stacked = self.stack_data(X)
        lle_all_points_clusters = self.smoothen_clusters(X_stacked, self.history_)

        # Update cluster points - using NEW smoothening
        return self._update_clusters(lle_all_points_clusters, switch_penalty=self.switch_penalty)

    def stack_data(self, X):
        # re-implemented
        n_samples, n_features = X.shape
        # n_features_stacked = n_features * self.window_size
        # X_stacked = np.concatenate([
        #     x.reshape(1, -1) if (x := X[start:start + self.window_size].ravel()).shape[0] == n_features_stacked else
        #     np.pad(x, (0, n_features_stacked - len(x)), 'constant', constant_values=0).reshape(1, -1)
        #     for start in range(0, n_samples)
        # ])
        X_stacked = np.concatenate([X[start:start + self.window_size].reshape(1, -1)
                                    for start in range(0, n_samples - self.window_size + 1)])

        return X_stacked

    def get_clustered_points_group(self, clustered_points):
        return {cluster: np.flatnonzero([clustered_points == cluster]) for cluster in range(self.n_clusters)}

    def train_clusters(self, X_stacked, history,):
        n_features_stacked = X_stacked.shape[1]
        n_features = n_features_stacked // self.window_size

        solvers = {}
        for cluster, indices in history['clustered_points_group'].items():
            if (cluster_length := len(indices)) != 0:
                block_size = n_features
                x_cluster = np.zeros([cluster_length, n_features_stacked])
                for i in range(cluster_length):
                    point = indices[i]
                    x_cluster[i, :] = X_stacked[point, :]

                # Fit a model - OPTIMIZATION
                prob_size = self.window_size * block_size
                lamb = np.zeros((prob_size, prob_size)) + self.lambda_parameter
                S = np.cov(x_cluster, bias=self.biased, rowvar=False)

                history['cluster_mean_stacked'][cluster] = np.mean(x_cluster, axis=0)
                history['cluster_mean'][cluster] = history['cluster_mean_stacked'][cluster][-n_features:].reshape(1, -1)
                history['empirical_covariances'][cluster] = S

                rho = 1
                solvers.setdefault(cluster, ADMM(
                    lamb=lamb,
                    num_stacked=self.window_size,
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

        opt_res = {k: v for k, v in zip(solvers, res)}
        history['opt_res'] = opt_res

    def optimize_clusters(self, history):
        # opt_res contains only results for clusters with length != 0

        for cluster, val in history['opt_res'].items():
            # THIS IS THE SOLUTION
            S_est = upper_to_full(val, 0)
            X2 = S_est
            u, _ = np.linalg.eig(S_est)
            cov_out = np.linalg.inv(X2)

            # Store the log-det, covariance, inverse-covariance, cluster means, stacked means
            history['log_det_values'][cluster] = np.log(np.linalg.det(cov_out))
            history['computed_covariance'][cluster] = cov_out
            history['train_cluster_inverse'][cluster] = X2

    def smoothen_clusters(self, X_stacked, history):
        n_samples, n_features_stacked = X_stacked.shape

        inv_cov_dict = {}  # cluster to inv_cov
        log_det_dict = {}  # cluster to log_det

        for cluster in range(self.n_clusters):
            cov_matrix = history['computed_covariance'][cluster]
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            log_det_cov = np.log(np.linalg.det(cov_matrix))  # log(det(sigma2|1))
            inv_cov_dict[cluster] = inv_cov_matrix
            log_det_dict[cluster] = log_det_cov
        # For each point compute the LLE
        print("beginning the smoothening ALGORITHM")

        lle_all_points_clusters = np.zeros(shape=(n_samples, self.n_clusters))
        for point in range(n_samples):
            if point + self.window_size - 1 < X_stacked.shape[0]:
                for cluster in range(self.n_clusters):
                    cluster_mean_stacked = history['cluster_mean_stacked'][cluster]
                    x = (X_stacked[point, :] - cluster_mean_stacked).reshape(-1, 1)
                    inv_cov_matrix = inv_cov_dict[cluster]
                    log_det_cov = log_det_dict[cluster]
                    lle = x.T @ inv_cov_matrix @ x + log_det_cov
                    lle_all_points_clusters[point, cluster] = lle

        return lle_all_points_clusters

    def update_clusters(self, X_stacked, history):
        '''
        Given the current trained model, predict clusters.  If the cluster segmentation has not been optimized yet,
        than this will be part of the interative process.

        Returns:
            vector of predicted cluster for the points
        '''

        # SMOOTHENING
        lle_all_points_clusters = self.smoothen_clusters(X_stacked, history)

        # Update cluster points - using NEW smoothening
        history['clustered_points'] = self._update_clusters(lle_all_points_clusters, switch_penalty=self.switch_penalty)

    @staticmethod
    def _update_clusters(lle_node_vals, switch_penalty=1):
        (T, num_clusters) = lle_node_vals.shape
        future_cost_vals = np.zeros(lle_node_vals.shape)

        # compute future costs
        for i in range(T - 2, -1, -1):
            j = i + 1
            future_costs = future_cost_vals[j, :]
            lle_vals = lle_node_vals[j, :]
            for cluster in range(num_clusters):
                total_vals = future_costs + lle_vals + switch_penalty
                total_vals[cluster] -= switch_penalty
                future_cost_vals[i, cluster] = np.min(total_vals)

        # compute the best path
        path = np.zeros(T)

        # the first location
        curr_location = np.argmin(future_cost_vals[0, :] + lle_node_vals[0, :])
        path[0] = curr_location

        # compute the path
        for i in range(T - 1):
            j = i + 1
            future_costs = future_cost_vals[j, :]
            lle_vals = lle_node_vals[j, :]
            total_vals = future_costs + lle_vals + switch_penalty
            total_vals[int(path[i])] -= switch_penalty

            path[i + 1] = np.argmin(total_vals)

        # return the computed path
        return path.astype(np.int32)

    def reassign_empty_clusters(self, X_stacked, history):
        n_features_stacked = X_stacked.shape[1]
        n_features = n_features_stacked // self.window_size

        cluster_norms = [(np.linalg.norm(cov), cluster)
                         for cluster, cov in history['computed_covariance'].items()]

        norms_sorted = sorted(cluster_norms, reverse=True)

        # clusters that are not 0 as sorted by norm
        valid_clusters = [cluster for norm, cluster in norms_sorted
                          if history['clustered_points_group'][cluster].size]

        # Add a point to the empty clusters
        # assuming more non empty clusters than empty ones
        counter = 0
        for cluster in range(self.n_clusters):
            if not history['clustered_points_group'][cluster].size:
                cluster_selected = valid_clusters[counter]  # a cluster that is not len 0
                counter = (counter + 1) % len(valid_clusters)
                print("cluster that is zero is:", cluster, "selected cluster instead is:", cluster_selected)

                # random point number from that cluster
                start_point = np.random.choice(history['clustered_points_group'][cluster_selected])
                for i in range(self.cluster_reassignment):
                    # put cluster_reassignment points from point_num in this cluster
                    point_to_move = start_point + i
                    if point_to_move >= len(history['clustered_points']):
                        break

                    history['clustered_points'][point_to_move] = cluster
                    history['clustered_points_group'][cluster] = np.flatnonzero([history['clustered_points'] == cluster])
                    history['computed_covariance'][cluster] = history['computed_covariance'][cluster_selected]
                    history['cluster_mean_stacked'][cluster] = X_stacked[point_to_move, :]
                    history['cluster_mean'][cluster] = X_stacked[point_to_move, -n_features:]

    def log_parameters(self):
        print("lam_sparse", self.lambda_parameter)
        print("switch_penalty", self.switch_penalty)
        print("num_cluster", self.n_clusters)
        print("num stacked", self.window_size)
