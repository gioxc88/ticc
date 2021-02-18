from collections import defaultdict

import numpy as np
import pandas as pd

from joblib import delayed, Parallel
from sklearn import mixture
from sklearn.base import BaseEstimator

from .utils import (
    upper_to_full,
    get_train_test_split,
    compute_bic,
    compute_confusion_matrix,
    find_matching,
    update_clusters
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

        # Get data into proper format
        times_series_arr = X
        time_series_rows_size, time_series_col_size = times_series_arr.shape

        # Train test split
        # indices of the training samples
        training_indices = get_train_test_split(time_series_rows_size, self.num_blocks, self.window_size)
        num_train_points = len(training_indices)

        # Stack the training data
        complete_D_train = self.stack_training_data(
            times_series_arr,
            num_train_points,
            training_indices
        )

        # Initialization
        # Gaussian Mixture
        gmm = mixture.GaussianMixture(n_components=self.n_clusters, covariance_type="full")
        gmm.fit(complete_D_train)
        clustered_points = gmm.predict(complete_D_train)
        gmm_clustered_pts = clustered_points + 0

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = None  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        for iter in range(self.max_iters):

            print("\n\n\nITERATION ###", iter)
            # Get the train and test points
            train_clusters_arr = defaultdict(list)  # {cluster: [point indices]}
            for point, cluster_num in enumerate(clustered_points):
                train_clusters_arr[cluster_num].append(point)

            len_train_clusters = {k: len(v) for k, v in train_clusters_arr.items()}

            # train_clusters holds the indices in complete_D_train
            # for each of the clusters
            opt_res = self.train_clusters(
                cluster_mean_info,
                cluster_mean_stacked_info,
                complete_D_train,
                empirical_covariances,
                len_train_clusters,
                time_series_col_size,
                train_clusters_arr
            )

            self.optimize_clusters(
                computed_covariance,
                len_train_clusters,
                log_det_values,
                opt_res,
                train_cluster_inverse
            )

            # update old computed covariance
            old_computed_covariance = computed_covariance

            print("UPDATED THE OLD COVARIANCE")

            self.trained_model = {
                'cluster_mean_info': cluster_mean_info,
                'computed_covariance': computed_covariance,
                'cluster_mean_stacked_info': cluster_mean_stacked_info,
                'complete_D_train': complete_D_train,
                'time_series_col_size': time_series_col_size
            }
            clustered_points = self.predict_clusters()

            # recalculate lengths
            new_train_clusters = defaultdict(list)  # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {k: len(new_train_clusters[k]) for k in range(self.n_clusters)}
            before_empty_cluster_assign = clustered_points.copy()

            if iter != 0:
                cluster_norms = [(np.linalg.norm(old_computed_covariance[self.n_clusters, i]), i) for i in
                                 range(self.n_clusters)]
                norms_sorted = sorted(cluster_norms, reverse=True)
                # clusters that are not 0 as sorted by norm
                valid_clusters = [cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

                # Add a point to the empty clusters
                # assuming more non empty clusters than empty ones
                counter = 0
                for cluster_num in range(self.n_clusters):
                    if len_new_train_clusters[cluster_num] == 0:
                        cluster_selected = valid_clusters[counter]  # a cluster that is not len 0
                        counter = (counter + 1) % len(valid_clusters)
                        print("cluster that is zero is:", cluster_num, "selected cluster instead is:", cluster_selected)
                        start_point = np.random.choice(
                            new_train_clusters[cluster_selected])  # random point number from that cluster
                        for i in range(0, self.cluster_reassignment):
                            # put cluster_reassignment points from point_num in this cluster
                            point_to_move = start_point + i
                            if point_to_move >= len(clustered_points):
                                break
                            clustered_points[point_to_move] = cluster_num
                            computed_covariance[self.n_clusters, cluster_num] = old_computed_covariance[
                                self.n_clusters, cluster_selected]
                            cluster_mean_stacked_info[self.n_clusters, cluster_num] = complete_D_train[
                                                                                      point_to_move, :]
                            cluster_mean_info[self.n_clusters, cluster_num] \
                                = complete_D_train[point_to_move, :][
                                  (self.window_size - 1) * time_series_col_size:self.window_size * time_series_col_size]

            for cluster_num in range(self.n_clusters):
                print("length of cluster #", cluster_num, "-------->",
                      sum([x == cluster_num for x in clustered_points]))

            if np.array_equal(old_clustered_points, clustered_points):
                print("\n\n\n\nCONVERGED!!! BREAKING EARLY!!!")
                break
            old_clustered_points = before_empty_cluster_assign
            # end of training

        if self.compute_bic:
            self.bic_ = compute_bic(
                time_series_rows_size,
                clustered_points,
                train_cluster_inverse,
                empirical_covariances
            )

        self.clustered_points_ = clustered_points
        self.train_cluster_inverse_ = train_cluster_inverse
        self.train_confusion_matrix_ = compute_confusion_matrix(self.n_clusters, clustered_points, training_indices)

        return self

    def compute_f_score(self, matching, train_confusion_matrix):
        # doesn't return anything?
        f1_tr = -1  # compute_f1_macro(train_confusion_matrix, matching, num_clusters)

        print("\n\n")
        print("TRAINING F1 score:", f1_tr)
        correct = 0
        for cluster in range(self.n_clusters):
            matched_cluster = matching[cluster]
            correct += train_confusion_matrix[cluster, matched_cluster]

    def compute_matches(self, train_confusion_matrix):
        matching = find_matching(train_confusion_matrix)
        correct = 0
        for cluster in range(self.n_clusters):
            matched_cluster = matching[cluster]
            correct += train_confusion_matrix[cluster, matched_cluster]
        return matching

    def smoothen_clusters(self, cluster_mean_info, computed_covariance, cluster_mean_stacked_info, complete_D_train, n):
        clustered_points_len = len(complete_D_train)
        inv_cov_dict = {}  # cluster to inv_cov
        log_det_dict = {}  # cluster to log_det
        for cluster in range(self.n_clusters):
            cov_matrix = computed_covariance[self.n_clusters, cluster][0:(self.num_blocks - 1) * n,
                         0:(self.num_blocks - 1) * n]
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            log_det_cov = np.log(np.linalg.det(cov_matrix))  # log(det(sigma2|1))
            inv_cov_dict[cluster] = inv_cov_matrix
            log_det_dict[cluster] = log_det_cov
        # For each point compute the LLE
        print("beginning the smoothening ALGORITHM")
        LLE_all_points_clusters = np.zeros([clustered_points_len, self.n_clusters])
        for point in range(clustered_points_len):
            if point + self.window_size - 1 < complete_D_train.shape[0]:
                for cluster in range(self.n_clusters):
                    cluster_mean = cluster_mean_info[self.n_clusters, cluster]
                    cluster_mean_stacked = cluster_mean_stacked_info[self.n_clusters, cluster]
                    x = complete_D_train[point, :] - cluster_mean_stacked[0:(self.num_blocks - 1) * n]
                    inv_cov_matrix = inv_cov_dict[cluster]
                    log_det_cov = log_det_dict[cluster]
                    lle = np.dot(x.reshape([1, (self.num_blocks - 1) * n]),
                                 np.dot(inv_cov_matrix, x.reshape([n * (self.num_blocks - 1), 1]))) + log_det_cov
                    LLE_all_points_clusters[point, cluster] = lle

        return LLE_all_points_clusters

    def optimize_clusters(
            self,
            computed_covariance,
            len_train_clusters,
            log_det_values,
            opt_res,
            train_cluster_inverse
    ):
        # opt_res contains only results for clusters with length != 0

        for cluster, val in opt_res.items():
            print("OPTIMIZATION for Cluster #", cluster, "DONE!!!")
            # THIS IS THE SOLUTION
            S_est = upper_to_full(val, 0)
            X2 = S_est
            u, _ = np.linalg.eig(S_est)
            cov_out = np.linalg.inv(X2)

            # Store the log-det, covariance, inverse-covariance, cluster means, stacked means
            log_det_values[self.n_clusters, cluster] = np.log(np.linalg.det(cov_out))
            computed_covariance[self.n_clusters, cluster] = cov_out
            train_cluster_inverse[cluster] = X2
        for cluster in range(self.n_clusters):
            print("length of the cluster ", cluster, "------>", len_train_clusters[cluster])

    def train_clusters(
            self,
            cluster_mean_info,
            cluster_mean_stacked_info,
            complete_D_train,
            empirical_covariances,
            len_train_clusters,
            n_features,
            train_clusters_arr
    ):

        solvers = {}
        for cluster in range(self.n_clusters):
            cluster_length = len_train_clusters[cluster]
            if cluster_length != 0:
                block_size = n_features
                indices = train_clusters_arr[cluster]
                D_train = np.zeros([cluster_length, self.window_size * n_features])
                for i in range(cluster_length):
                    point = indices[i]
                    D_train[i, :] = complete_D_train[point, :]

                cluster_mean_info[self.n_clusters, cluster] = \
                    np.mean(D_train, axis=0)[(self.window_size - 1) * n_features:
                                             self.window_size * n_features].reshape([1, n_features])
                cluster_mean_stacked_info[self.n_clusters, cluster] = np.mean(D_train, axis=0)

                # Fit a model - OPTIMIZATION
                prob_size = self.window_size * block_size
                lamb = np.zeros((prob_size, prob_size)) + self.lambda_parameter
                S = np.cov(D_train, bias=self.biased, rowvar=False)
                empirical_covariances[cluster] = S

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
        return opt_res

    def stack_training_data(self, X, num_train_points, training_indices):
        n = X.shape[1]
        complete_D_train = np.zeros([num_train_points, self.window_size * n])
        for i in range(num_train_points):
            for k in range(self.window_size):
                if i + k < num_train_points:
                    idx_k = training_indices[i + k]
                    complete_D_train[i][k * n:(k + 1) * n] = X[idx_k][0:n]
        return complete_D_train

    def log_parameters(self):
        print("lam_sparse", self.lambda_parameter)
        print("switch_penalty", self.switch_penalty)
        print("num_cluster", self.n_clusters)
        print("num stacked", self.window_size)

    def predict_clusters(self, test_data=None):
        '''
        Given the current trained model, predict clusters.  If the cluster segmentation has not been optimized yet,
        than this will be part of the interative process.

        Args:
            numpy array of data for which to predict clusters.  Columns are dimensions of the data, each row is
            a different timestamp

        Returns:
            vector of predicted cluster for the points
        '''
        if test_data is not None:
            if not isinstance(test_data, np.ndarray):
                raise TypeError("input must be a numpy array!")
        else:
            test_data = self.trained_model['complete_D_train']

        # SMOOTHENING
        lle_all_points_clusters = self.smoothen_clusters(
            self.trained_model['cluster_mean_info'],
            self.trained_model['computed_covariance'],
            self.trained_model['cluster_mean_stacked_info'],
            test_data,
            self.trained_model['time_series_col_size']
        )

        # Update cluster points - using NEW smoothening
        clustered_points = update_clusters(lle_all_points_clusters, switch_penalty=self.switch_penalty)

        return clustered_points
