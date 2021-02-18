import numpy as np
from scipy import stats


def get_train_test_split(m, num_blocks, num_stacked):
    '''
    - m: number of observations
    - num_blocks: window_size + 1
    - num_stacked: window_size
    Returns:
    - sorted list of training indices
    '''
    # Now splitting up stuff
    # split1 : Training and Test
    # split2 : Training and Test - different clusters
    training_percent = 1
    # list of training indices
    training_idx = np.random.choice(
        m - num_blocks + 1, size=int((m - num_stacked) * training_percent), replace=False)
    # Ensure that the first and the last few points are in
    training_idx = list(training_idx)
    if 0 not in training_idx:
        training_idx.append(0)
    if m - num_stacked not in training_idx:
        training_idx.append(m - num_stacked)
    training_idx = np.array(training_idx)
    return sorted(training_idx)


def upper_to_full(a, eps=0):
    if eps is not None:
        mask = (a < eps) & (a > -eps)
        a[mask] = 0
    n = int((-1 + np.sqrt(1 + 8 * a.shape[0])) / 2)
    A = np.zeros([n, n])
    A[np.triu_indices(n)] = a
    temp = A.diagonal()
    A = np.asarray((A + A.T) - np.diag(temp))
    return A


def hex_to_rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    lv = len(value)
    out = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    out = tuple([x / 256.0 for x in out])
    return out


def update_clusters(lle_node_vals, switch_penalty=1):
    """
    Takes in lle_node_vals matrix and computes the path that minimizes
    the total cost over the path
    Note the lle's are negative of the true lle's actually!!!!!

    Note: switch penalty > 0
    """
    (T, num_clusters) = lle_node_vals.shape
    future_cost_vals = np.zeros(lle_node_vals.shape)

    # compute future costs
    for i in range(T - 2, -1, -1):
        j = i + 1
        indicator = np.zeros(num_clusters)
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
    return path


def find_matching(confusion_matrix):
    """
    returns the perfect matching
    """
    _, n = confusion_matrix.shape
    path = []
    for i in range(n):
        max_val = -1e10
        max_ind = -1
        for j in range(n):
            if j in path:
                pass
            else:
                temp = confusion_matrix[i, j]
                if temp > max_val:
                    max_val = temp
                    max_ind = j
        path.append(max_ind)
    return path


def compute_confusion_matrix(num_clusters, clustered_points_algo, sorted_indices_algo):
    """
    computes a confusion matrix and returns it
    """
    seg_len = 400
    true_confusion_matrix = np.zeros([num_clusters, num_clusters])
    for point in range(len(clustered_points_algo)):
        cluster = clustered_points_algo[point]
        num = (int(sorted_indices_algo[point] / seg_len) % num_clusters)
        true_confusion_matrix[int(num), int(cluster)] += 1
    return true_confusion_matrix


def compute_f1_macro(confusion_matrix, matching, num_clusters):
    """
    computes the macro F1 score
    confusion matrix : requires permutation
    matching according to which matrix must be permuted
    """
    # Permute the matrix columns
    permuted_confusion_matrix = np.zeros([num_clusters, num_clusters])
    for cluster in range(num_clusters):
        matched_cluster = matching[cluster]
        permuted_confusion_matrix[:, cluster] = confusion_matrix[:, matched_cluster]
    # Compute the F1 score for every cluster
    F1_score = 0
    for cluster in range(num_clusters):
        TP = permuted_confusion_matrix[cluster, cluster]
        FP = np.sum(permuted_confusion_matrix[:, cluster]) - TP
        FN = np.sum(permuted_confusion_matrix[cluster, :]) - TP
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = stats.hmean([precision, recall])
        F1_score += f1
    F1_score /= num_clusters
    return F1_score


def compute_bic(t, clustered_points, inverse_covariances, empirical_covariances):
    '''
    empirical covariance and inverse_covariance should be dicts
    t is num samples
    '''
    mod_lle = 0

    threshold = 2e-5
    cluster_params = {}
    for cluster, cluster_inverse in inverse_covariances.items():
        mod_lle += np.log(np.linalg.det(cluster_inverse)) - np.trace(
            np.dot(empirical_covariances[cluster], cluster_inverse))
        cluster_params[cluster] = np.sum(np.abs(cluster_inverse) > threshold)
    curr_val = -1
    non_zero_params = 0
    for val in clustered_points:
        if val != curr_val:
            non_zero_params += cluster_params[val]
            curr_val = val
    return non_zero_params * np.log(t) - 2 * mod_lle
