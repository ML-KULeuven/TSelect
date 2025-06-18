from collections import defaultdict
from typing import List, Union, Set

import numpy as np
from scipy.linalg import eigh
from scipy.stats import permutation_test, spearmanr
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors

from tselect.utils import Keys
from tselect.utils import average
from tselect.rank_correlation.spearman import standard_deviation, spearman, spearman_distinct_ranks

from tselect.utils.constants import SEED


def probabilities2rank(probabilities: np.ndarray) -> np.ndarray:
    """
    Ranks an array of probabilities and converts it to an array of ranks. A rank of 1 corresponds to the highest
    probability. If the given array is 2-dimensional, every column gets ranked separately.

    Parameters
    ----------
    probabilities : np.ndarray
        The array of probabilities to convert to ranks.

    Returns
    -------
    np.ndarray
        The array of ranks.
    """
    if len(probabilities.shape) == 1:
        return probabilities2rank1d(probabilities)
    else:
        result = np.empty(shape=probabilities.shape, dtype=int)
        for i in range(probabilities.shape[1]):
            result[:, i] = probabilities2rank1d(probabilities[:, i])
        return result


def probabilities2rank1d(probabilities: np.ndarray) -> np.ndarray:
    """
    Converts a 1D array of probabilities to a 1D array of ranks. A rank of 1 corresponds with the highest probability.

    Parameters
    ----------
    probabilities : np.ndarray
        The 1D array of probabilities to convert to ranks.

    Returns
    -------
    np.ndarray
        The 1D array of ranks.
    """
    indices = np.flip(np.argsort(probabilities))
    result = np.empty(shape=probabilities.shape, dtype=int)
    for i in range(1, len(probabilities) + 1):
        result[indices[i - 1]] = i
    return result


def pairwise_rank_correlation_old(ranks: dict, perm_test: bool = False) -> dict:
    """
    Computes the pairwise rank correlation between the elements in the input `ranks` dict.

    Parameters
    ----------
    ranks : dict
        A dictionary containing the rankings of the different features for the target variable(s). The keys identify
         where the ranking belongs to and the values correspond with the ranking of the probabilities.

    perm_test : bool, optional
        Whether to perform permutation test, by default False. If the number of instances is lower than 500, the p-value
        calculated by `spearmanr` is inaccurate and a permutation test should be used instead.

    Returns
    -------
    dict
        A dict containing the correlation for each pair of keys in the `ranks` dictionary.
    """
    columns = list(ranks.keys())
    result = {}
    binary = ranks[columns[0]].ndim == 1
    for i, col_i in enumerate(columns):
        for col_j in columns[i + 1:]:
            if binary:
                corr, pvalue = get_corr_pvalue(ranks[col_i], ranks[col_j], perm_test)
                result_corr = np.array([corr])
            else:
                result_corr = np.empty(shape=ranks[col_i].shape[1])
                for target_var in range(ranks[col_i].shape[1]):
                    corr, _ = get_corr_pvalue(ranks[col_i][:, target_var], ranks[col_j][:, target_var], perm_test)
                    result_corr[target_var] = corr

            result[(col_i, col_j)] = result_corr
    return result


def pairwise_rank_correlation(ranks: dict) -> dict:
    """
    Computes the pairwise rank correlation between the elements in the input `ranks` dict.

    Parameters
    ----------
    ranks : dict
        A dictionary containing the rankings of the different features for the target variable(s). The keys identify
         where the ranking belongs to and the values correspond with the ranking of the probabilities.

    Returns
    -------
    dict
        A dict containing the correlation for each pair of keys in the `ranks` dictionary.
    """
    channels = list(ranks.keys())
    result = {}
    std = compute_standard_devs(ranks)
    for i, channel_i in enumerate(channels):
        for channel_j in channels[i + 1:]:
            result[(channel_i, channel_j)] = (
                spearman(ranks[channel_i], ranks[channel_j], std[channel_i], std[channel_j]))
    return result


def pairwise_rank_correlation_opt(ranks: dict) -> (dict, Set):
    """
    Computes the pairwise rank correlation between the elements in the input `ranks` dict.

    Parameters
    ----------
    ranks : dict
        A dictionary containing the rankings of the different features for the target variable(s). The keys identify
         where the ranking belongs to and the values correspond with the ranking of the probabilities.

    Returns
    -------
    dict
        A dict containing the correlation for each pair of keys in the `ranks` dictionary.
    """
    channels = list(ranks.keys())
    result = {}
    for i, channel_i in enumerate(channels):
        for channel_j in channels[i + 1:]:
            result[(channel_i, channel_j)] = spearman_distinct_ranks(ranks[channel_i], ranks[channel_j])
    return result, set(channels)


def pairwise_rank_correlation_opt_early_stop(ranks: dict, sorted_auc: List[tuple], corr_threshold: float) -> (dict, Set):
    """
    Computes the pairwise rank correlation between the elements in the input `ranks` dict. The computation is optimized
    by skipping the computation of the correlation between two series if the correlation between the first-occurring
    series and later-occurring series is already above the threshold. As only the best-performing series are selected
    from each cluster, further computations are not necessary.

    Parameters
    ----------
    ranks : dict
        A dictionary containing the rankings of the different signals. The keys contain a signal and the values contain
        the ranking of the signal.
    sorted_auc : list of tuples
        A list of tuples containing the AUC score of each signal and the signal itself. The list is sorted in descending
        order of the AUC scores.
    corr_threshold : float
        The threshold to use to determine if two signals are correlated.

    Returns
    -------
    dict
        A dict containing the correlation for each pair of keys in the `ranks` dictionary.
    Set
        A set containing the signals that were included in the computation of the rank correlations. All skipped signals
        are not included in this set.

    """
    columns = list(ranks.keys())
    result = {}
    binary = ranks[columns[0]].ndim == 1
    suboptimal_series = set()
    included_series = set()

    def compute_rank_correlation(x, y):
        if binary:
            corr, _ = get_corr_pvalue(x, y, False)
            return np.array([corr])

        else:
            result_corr = np.empty(shape=x.shape[1])
            for target_var in range(x.shape[1]):
                corr, _ = get_corr_pvalue(x[:, target_var], y[:, target_var], False)
                result_corr[target_var] = corr
            return result_corr

    nb_called = 0
    for i, col_i in enumerate(sorted_auc):
        if col_i in suboptimal_series:
            continue
        included_series.add(col_i)
        for col_j in sorted_auc[i + 1:]:
            corr_ij = compute_rank_correlation(ranks[col_i], ranks[col_j])
            nb_called += 1
            if np.mean(np.abs(corr_ij)) >= corr_threshold:
                suboptimal_series.add(col_j)
            else:
                included_series.add(col_j)

            result[(col_i, col_j)] = corr_ij
    print("Only ", nb_called, " of the ", len(sorted_auc) * (len(sorted_auc) - 1) / 2, " combinations were calculated")
    return result, included_series


def compute_standard_devs(ranks: dict) -> dict:
    """
    Computes the standard deviation of the ranks for each signal and for each class.

    Parameters
    ----------
    ranks : dict
        A dictionary containing the rankings of the different signals. The keys contain a signal and the values contain
        the ranking of the signal.

    Returns
    -------
    dict
        A dictionary containing the standard deviation of the ranks for each signal. The value of each dictionary is an
        array containing the standard deviation for each class.
    """
    result = {}
    for key, value in ranks.items():
        result[key] = standard_deviation(value)
    return result


def get_corr_pvalue(x, y, perm_test):
    """
    Compute the rank correlation and p-value between `x` and `y`.

    Parameters
    ----------
    x: 1D array-like of ints
        the first array containing ranks
    y: 1D array-like of ints
        the second array containing ranks
    perm_test: bool
        boolean indicating whether the permutation test should be used to compute the p-value. If the number of
        instances is lower than 500, the p-value calculated by `spearmanr` is inaccurate and a permutation test should
        be used instead.

    Returns
    -------
    corr: float or array of floats
        the correlation between `x` and `y`, as computed by the `spearmanr` function
    pvalue: float or array of floats
        the p-value between `x` and `y`. If perm_test is True, this is computed by using a perumation test. If not,
        the p-value is computed by the `spearmanr` function.
    """
    if perm_test:
        res = permutation_test((x,), lambda xi: spearmanr(xi, y)[0], permutation_type='pairings')
        return res.statistic, res.pvalue
    else:
        corr, pvalue = spearmanr(x, y, axis=0)
        return corr, pvalue


def cluster_correlations(rank_correlations: dict, included_series: Set = None, threshold: float = 0.7, optimized=False) \
        -> List[List[Union[str, int]]]:
    """
    Clusters the signals based on their rank correlations. Each cluster contains signals that are highly correlated to
    each other.

    Parameters
    ----------
    rank_correlations: dict
        The rank correlations between the signals, with a tuple of the signals as key and the correlation as value
    included_series: set, optional, default None
        The set of series of which the rank computations were computed. If no set is given, the set of series is
        assumed to be all series present in the rank_correlations dictionary.
    threshold: float, optional, default 0.7
        The threshold to use to determine if two signals are correlated.
    optimized: bool, optional, default False
        Whether to use the optimized version for clustering. Although this version is faster, it is not guaranteed to
        find the optimal clustering.

    Returns
    -------
    List[set]
        A list of sets, where each set represents a cluster, containing the signals that are highly correlated to each
        other.
    """
    if included_series is None:
        included_series = set()
        for (s1, s2) in rank_correlations.keys():
            included_series.add(s1)
            included_series.add(s2)
    clusters: List[list] = []
    unallocated_set = included_series.copy()
    for (s1, s2) in rank_correlations.keys():
        # If the first key should be included, but the second key was skipped during the computation of the rank
        # correlations, continue. The first signal will get added (if not already) and the second one should be left
        # out.
        if s1 in included_series and s2 not in included_series:
            continue

        corr = np.mean(np.abs(rank_correlations[(s1, s2)]))

        # Scenario 1: the pair of series are not correlated
        if corr < threshold:
            continue

        # Scenario 2: the pair of series are correlated and both are not allocated yet
        elif s1 in unallocated_set and s2 in unallocated_set:
            # clusters.append({s1, s2})
            clusters.append([s1, s2])
            unallocated_set.remove(s1)
            unallocated_set.remove(s2)
            # print("New cluster, there are now ", len(clusters), " clusters")

        # Scenario 3: the pair of series are correlated and both are already allocated
        elif s1 not in unallocated_set and s2 not in unallocated_set:
            cluster1 = [c for c in clusters if s1 in c][0]

            # Scenario 3.1: both series are already in the same cluster
            if s2 in cluster1:
                continue

            # Scenario 3.2: both series are in different clusters -> transform clusters to maximize intra cluster
            # correlation
            cluster2 = [c for c in clusters if s2 in c][0]
            new_cluster1, new_cluster2 = transform_clusters(cluster1, cluster2, rank_correlations, threshold)
            clusters.remove(cluster1)
            clusters.remove(cluster2)
            if len(new_cluster1) > 1:
                clusters.append(new_cluster1)
                # print("Adapted cluster: size changed from ", len(cluster1), " to ", len(new_cluster1))
            else:
                unallocated_set.update(set(new_cluster1))
                # print("Removed one cluster, there are now ", len(clusters), " clusters")
            if len(new_cluster2) > 1:
                clusters.append(new_cluster2)
                # print("Adapted cluster: size changed from ", len(cluster2), " to ", len(new_cluster2))
            else:
                unallocated_set.update(set(new_cluster2))
                # print("Removed one cluster, there are now ", len(clusters), " clusters")

        # Scenario 4: the pair of series are correlated and one of the series is already allocated, but the other is not
        else:
            allocated_s = s1 if s1 not in unallocated_set else s2
            other_s = s1 if s1 in unallocated_set else s2
            cluster_ix = [i for i in range(len(clusters)) if allocated_s in clusters[i]][0]

            # if optimized:
            #     clusters[cluster_ix].add(other_s)
            #     unallocated_set.remove(other_s)
            #     continue
            correlated, corr = check_correlated(other_s, list(clusters[cluster_ix]), rank_correlations, threshold)

            # Scenario 4.1: other series highly correlated to all elements in the cluster -> add it
            if all(correlated):
                # clusters[cluster_ix].add(other_s)
                clusters[cluster_ix].append(other_s)
                unallocated_set.remove(other_s)
                # print("Adapted cluster: size changed incremented with 1 to", len(clusters[cluster_ix]))

            # Scenario 4.2: other series is not correlated to some elements in the cluster -> split cluster to maximize
            # intra cluster correlation
            else:
                cluster1, cluster2 = split_cluster_series(other_s, clusters[cluster_ix], rank_correlations, correlated)
                del clusters[cluster_ix]
                if len(cluster1) > 1:
                    clusters.append(cluster1)
                    unallocated_set = unallocated_set.difference(set(cluster1))
                    # print("Adapted cluster by splitting: size changed from ", len(clusters[cluster_ix]), " to ", len(cluster1))
                else:
                    unallocated_set.update(set(cluster1))
                    # print("Removed one cluster, there are now ", len(clusters), " clusters")
                if len(cluster2) > 1:
                    clusters.append(cluster2)
                    unallocated_set = unallocated_set.difference(set(cluster2))
                    # print("Adapted cluster by splitting: size changed from ", len(clusters[cluster_ix]), " to ", len(cluster2))
                else:
                    unallocated_set.update(set(cluster2))
                    # print("Removed one cluster, there are now ", len(clusters), " clusters")

    # Add unallocated series as separate clusters
    clusters.extend([[x] for x in unallocated_set])

    return clusters


def split_cluster_series(series: Union[int, str], to_split: list, rank_correlations: dict, correlated: List[bool]) \
        -> (list, list):
    """
    Adds a series to a cluster of signals. The given series is assumed to not be correlated to at least one of the
    signals in the cluster. The cluster is split in a greedy way into two new clusters such that the correlation
    between the signals of one cluster is maximized in a greedy, non-optimal way.
    Parameters
    ----------
    series: str or int
        The series to split the cluster on.
    to_split: set
        The cluster to split.
    rank_correlations: dict
        The rank correlations between the signals, with a tuple of the signals as key and the correlation as value
    correlated: list of bool
        A list of booleans indicating whether the series is correlated to the corresponding series in the cluster.

    Returns
    -------
    set
        The first new cluster.
    set
        The second new cluster.
    """
    cluster1 = [k for i, k in enumerate(to_split) if not correlated[i]]
    cluster2 = [series]

    split_cluster_cluster(cluster1, cluster2, to_split, rank_correlations)
    assert len(cluster1) + len(cluster2) == len(to_split) + 1
    return cluster1, cluster2


def split_cluster_cluster(cluster1: list, cluster2: list, to_split: list, rank_correlations):
    """
    Splits a cluster of signals into two new clusters such that the correlation between the signals of one cluster is
    maximized in a greedy, non-optimal way. The start for the two new clusters is given by the given clusters. If these
    are empty, correct behaviour cannot be guaranteed.

    Parameters
    ----------
    cluster1: set
        The beginning of the first cluster. This set cannot be empty.
    cluster2: set
        The beginning of the second cluster. This set cannot be empty.
    to_split: set
        The cluster to split.
    rank_correlations: dict
        The rank correlations between the signals, with a tuple of the signals as key and the correlation as value
    """

    def get_mean_corr(x, cluster):
        corr = []
        for c in cluster:
            key = (x, c) if (x, c) in rank_correlations else (c, x)
            if key in rank_correlations:
                corr.append(np.mean(np.abs(rank_correlations[key])))
        return average(corr)

    for s in to_split:
        if s in cluster1:
            continue
        corr1 = get_mean_corr(s, cluster1)
        corr2 = get_mean_corr(s, cluster2)
        if corr1 > corr2:
            # cluster1.add(s)
            cluster1.append(s)
        else:
            # cluster2.add(s)
            cluster2.append(s)


def transform_clusters(cluster1: list, cluster2: list, rank_correlations: dict, threshold: float) -> (list, list):
    """
    Transforms two clusters into two new clusters. If all signals in the two clusters are correlated, the clusters are
    merged. If not, the clusters are transformed into two new clusters such that the correlation between the signals of
    one cluster is maximized.

    Parameters
    ----------
    cluster1: set
        The first cluster
    cluster2: set
        The second cluster
    rank_correlations: dict
        The rank correlations between the signals, with a tuple of the signals as key and the correlation as value
    threshold: float
        The threshold for the correlation between two signals to be considered correlated
    """
    uncorrelated1, _ = find_uncorrelated_signals(cluster1, cluster2, rank_correlations, threshold)

    # If there are no uncorrelated signals, the clusters should be merged
    if uncorrelated1 is None:
        # return cluster1 | cluster2, set()
        return cluster1 + cluster2, []
    new_cluster1 = [uncorrelated1]

    # Make a new cluster with all uncorrelated signals of the uncorrelated signal of cluster1
    correlated, _ = check_correlated(uncorrelated1, list(cluster2), rank_correlations, threshold)
    new_cluster2 = [k for i, k in enumerate(cluster2) if not correlated[i]]

    # Everything else that is not already in the new clusters should be split
    # to_split = [k for k in (cluster2 | cluster1) if k not in new_cluster2 and k != uncorrelated1]
    to_split = [k for k in (cluster2 + cluster1) if k not in new_cluster2 and k != uncorrelated1]

    split_cluster_cluster(new_cluster1, new_cluster2, to_split, rank_correlations)
    assert len(cluster1) + len(cluster2) == len(new_cluster1) + len(new_cluster2)
    return new_cluster1, new_cluster2


def find_uncorrelated_signals(cluster1: list, cluster2: list, rank_correlations: dict, threshold: float) -> (str, str):
    """
    Find a pair of signals that are uncorrelated to each other. If no such pair exists, return None.

    Parameters
    ----------
    cluster1: set of str or int
        The first cluster of signals.
    cluster2: set of str or int
        The second cluster of signals.
    rank_correlations: dict
        The dictionary of rank correlations containing the correlation values of all pairs of series.
    threshold: float
        The threshold to use to determine if two series are correlated.

    Returns
    -------
    str, str
        The pair of uncorrelated signals.
    """
    for s1 in cluster1:
        for s2 in cluster2:
            correlated, corr = check_correlated(s1, [s2], rank_correlations, threshold)
            if not correlated[0]:
                return s1, s2
    return None, None


def check_correlated(test_series: Union[str, int], corr_series: List[Union[str, int]], rank_correlations: dict,
                     threshold: float = 0.5) -> (List[bool], List[float]):
    """
    Check if a test series is correlated to a list of series. If the mean correlation (over all classification classes)
    of a pair of series is greater than or equal to the threshold, the pair is considered correlated. If the pair of
    series is not present in the rank_correlations dictionary, the pair is considered correlated (True is added at the
    corresponding index in the result list). However, no correlation value is added to the corrs list for this signal.

    Parameters
    ----------
    test_series: str or int
        The series to test for correlation.
    corr_series: list of str or int
        The list of series to test against.
    rank_correlations: dict
        The dictionary of rank correlations containing the correlation values of all pairs of series.
    threshold: float
        The threshold for correlation. If the mean correlation (over all classification classes) of a pair of series is
        greater than or equal to this threshold, the pair is considered correlated.

    Returns
    -------
    result: list of bool
        A list of booleans indicating whether the test series is correlated to the corresponding series in the list.
    corrs: list of float
        A list of mean correlation values of the test series and each series in the list.
    """
    result = []
    corrs = []
    for s in corr_series:
        key = (test_series, s) if (test_series, s) in rank_correlations else (s, test_series)
        if key not in rank_correlations:
            # This pair of columns was not added to rank correlations because one of the columns would never get chosen
            result.append(True)
            continue

        mean_corr = np.mean(np.abs(rank_correlations[key]))
        if mean_corr >= threshold:
            result.append(True)
        else:
            result.append(False)
        corrs.append(mean_corr)
    return result, corrs


def average_rank_corr_folds(rank_correlations: List[dict]) -> dict:
    """
    Averages the rank correlations over all folds and target variables. The average rank correlation is computed by
    taking the absolute value of the rank correlation and averaging over all folds and target variables.

    Parameters
    ----------
    rank_correlations: list of dict
        A list of dictionaries containing the rank correlations for each fold and target variable.

    Returns
    -------
    dict
        A dictionary containing the average rank correlation for each pair of signals.

    """
    result = {}
    for rank_corr in rank_correlations:
        for key, values in rank_corr.items():
            if key in result:
                result[key] = np.add(result[key], abs(values))
            elif reversed(key) in result:
                result[reversed(key)] = np.add(result[reversed(key)],
                                               abs(values))
            else:
                result[key] = values
                result[key] = abs(result[key])

    all_average_corrs = []

    all_corrs = {}
    nb_folds = len(rank_correlations)
    for i, (key, values) in enumerate(result.items()):
        result[key] = values / nb_folds

        for iv, v in enumerate(result[key]):
            if iv in all_corrs:
                all_corrs[iv].append(v)
            else:
                all_corrs[iv] = [v]

        average_corr = np.mean(result[key])
        result[key][Keys.average_correlation_target_variables] = average_corr
        all_average_corrs.append(average_corr)

    result[Keys.average_correlation_all] = average(all_average_corrs)
    result[Keys.max_correlation_all] = max(all_average_corrs)
    result[Keys.min_correlation_all] = min(all_average_corrs)

    return result

def cluster_correlations_agglo_hierarchical(rank_correlations: dict, included_series: Set = None,
                                            correlation_threshold: float = 0.7) \
        -> List[List[Union[str, int]]]:
    """
    Clusters the signals based on their rank correlations using a hierarchical clustering algorithm.
    Each cluster contains signals that are highly correlated to each other.

    Parameters
    ----------
    rank_correlations: dict
        The rank correlations between the signals, with a tuple of the signals as key and the correlation as value
    included_series: set, optional, default None
        The set of series of which the rank computations were computed. If no set is given, the set of series is
        assumed to be all series present in the rank_correlations dictionary.
    correlation_threshold: float, optional, default 0.7
        The threshold to use to determine if two signals are correlated.

    Returns
    -------
    List[set]
        A list of sets, where each set represents a cluster, containing the signals that are highly correlated to each
        other.
    """
    import scipy.cluster.hierarchy as sch
    import scipy.spatial.distance as ssd
    if included_series is None:
        included_series = set()
        for (s1, s2) in rank_correlations.keys():
            included_series.add(s1)
            included_series.add(s2)

    channels = sorted(included_series)  # Ensure a consistent order
    channels_index = {ch: i for i, ch in enumerate(channels)}

    # Initialize distance matrix
    n = len(channels)
    distance_matrix = np.ones((n, n))  # Start with 1s (max distance)

    # Fill distance matrix using the correlation dictionary
    for (ch1, ch2), corr in rank_correlations.items():
        i, j = channels_index[ch1], channels_index[ch2]
        dist = 1 - np.mean(np.abs(rank_correlations[(ch1, ch2)]))  # Convert correlation to distance
        distance_matrix[i, j] = distance_matrix[j, i] = dist  # Symmetric

    # Convert to condensed distance matrix format
    condensed_dist_matrix = ssd.squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering
    linkage_matrix = sch.linkage(condensed_dist_matrix, method='average')

    # Convert correlation threshold to distance threshold
    distance_threshold = 1 - correlation_threshold

    # Get cluster labels
    clusters = sch.fcluster(linkage_matrix, distance_threshold, criterion='distance')

    # Group features by cluster
    cluster_groups = defaultdict(list)
    for i, feature in enumerate(channels):
        cluster_groups[clusters[i]].append(feature)

    # Convert to list of lists
    return list(cluster_groups.values())


def cluster_correlations_spectral_clustering(rank_correlations: dict, included_series: Set = None, max_clusters=None,
                                             k_nn=7, random_state=SEED) \
        -> List[List[Union[str, int]]]:
    """
       Clusters the signals based on their rank correlations using a spectral clustering algorithm that uses a local
       scale and the eigengap heuristic to estimate the number of clusters. The local scale is based on the work of
       Zelnik-Manor, L., & Perona, P. (2004). Self-tuning spectral clustering. Advances in neural information processing
       systems, 17. The eigengap heuristic is based on the work of Von Luxburg, U. (2007). A tutorial on spectral
       clustering. Statistics and computing, 17, 395-416.

       Parameters
       ----------
       rank_correlations: dict
           The rank correlations between the signals, with a tuple of the signals as key and the correlation as value
       included_series: set, optional, default None
           The set of series of which the rank computations were computed. If no set is given, the set of series is
           assumed to be all series present in the rank_correlations dictionary.
       max_clusters: int, optional, default None
           The maximum number of clusters to use. If None, the number of clusters is set to the number of unique
           channels.
       k_nn: int, optional, default 7
           The number of nearest neighbors to use for computing the local scale. This parameter controls the
           locality of the clustering.
       random_state: int, optional, default SEED
            The random state to use for the clustering algorithm. This is used to ensure reproducibility of the
            clustering results.

       Returns
       -------
       List[set]
           A list of sets, where each set represents a cluster, containing the signals that are highly correlated to each
           other.
       """
    from scipy.sparse.csgraph import laplacian as csgraph_laplacian
    # Step 1: Extract all unique channels and index mapping
    if included_series is None:
        included_series = set()
        for (s1, s2) in rank_correlations.keys():
            included_series.add(s1)
            included_series.add(s2)

    # At least three channels are needed to perform spectral clustering with the eigengap heuristic
    if len(included_series) < 3:
        return [[ch] for ch in included_series]

    channels = sorted(included_series)  # Ensure deterministic ordering
    index_map = {feat: i for i, feat in enumerate(channels)}
    n = len(channels)
    if max_clusters is None:
        max_clusters = n

    # Step 2: Build distance matrix
    distance_matrix = np.ones((n, n))
    for (ch1, ch2), corr in rank_correlations.items():
        if ch1 not in included_series or ch2 not in included_series:
            continue
        i, j = index_map[ch1], index_map[ch2]
        dist = 1 - np.mean(np.abs(rank_correlations[(ch1, ch2)]))
        distance_matrix[i, j] = distance_matrix[j, i] = dist

    # Step 3: Compute local scales
    sigma = compute_local_scales(distance_matrix, k_nn)

    # Step 4: Build similarity matrix
    similarity_matrix = build_similarity_matrix(distance_matrix, sigma)

    # Step 5: Compute normalized Laplacian and eigenvalues
    L, D = csgraph_laplacian(csgraph=similarity_matrix, normed=True, return_diag=True)
    eigvals, eigvecs = eigh(L)  # full eigen decomposition (small n)

    # Step 6: Estimate number of clusters using eigengap
    eigvals = np.real(eigvals[:max_clusters + 1])
    gaps = np.diff(eigvals)
    est_k = np.argmax(gaps[1:]) + 1  # skip the first eigenvalue (always near 0)

    # Step 7: Run spectral clustering
    clustering = SpectralClustering(
        n_clusters=est_k,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=random_state
    )
    labels = clustering.fit_predict(similarity_matrix)

    # Group channels by cluster label
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(channels[i])

    return list(clusters.values())

def compute_local_scales(distance_matrix, k):
    """
    Compute local scale σ_i for each point as the distance to the k-th nearest neighbor.
    """
    n = distance_matrix.shape[0]
    k = min(k, n - 1)  # Ensure k is within bounds
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), metric='precomputed').fit(distance_matrix)
    distances, _ = nbrs.kneighbors(distance_matrix)
    sigma = distances[:, k]  # σᵢ = distance to k-th nearest neighbor
    return sigma

def build_similarity_matrix(distance_matrix, sigma):
    """
    Build locally-scaled similarity matrix.
    """
    n = distance_matrix.shape[0]
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                denom = sigma[i] * sigma[j]
                if denom > 0:
                    similarity_matrix[i, j] = np.exp(-distance_matrix[i, j] ** 2 / denom)
                else:
                    similarity_matrix[i, j] = 0.0
    return similarity_matrix
#
# def normalize_affinity_matrix(A):
#     """
#     Normalize the affinity matrix to create L = D^(-1/2) A D^(-1/2)
#     """
#     D = np.sum(A, axis=1)
#     D_inv_sqrt = np.diag(1.0 / np.sqrt(D + np.finfo(float).eps))  # Add epsilon to avoid div by 0
#     L = D_inv_sqrt @ A @ D_inv_sqrt
#     return L
#
# def best_rotation(Z):
#     """
#     Find the best orthogonal rotation matrix R that aligns Z with canonical axes using SVD.
#     """
#     _, _, Vt = np.linalg.svd(Z.T)
#     R = Vt.T
#     return R
#
# def alignment_cost(Z):
#     """
#     Compute alignment cost from paper:
#     J = sum_i (1 - max_j Z_ij^2)
#     """
#     max_squared = np.max(Z ** 2, axis=1)
#     return np.sum(1 - max_squared)
#
# def assign_clusters(Z):
#     """
#     Assign each point to the cluster with the maximum squared value in its rotated eigenvector.
#     """
#     return np.argmax(Z ** 2, axis=1)
#
# def self_tuning_spectral_clustering_rotation(rank_correlations, included_series, C_max=10, k=7):
#     """
#     Full Self-Tuning Spectral Clustering with automatic cluster number detection using rotation cost.
#
#     Parameters:
#     - correlation_dict: dict with keys (f1, f2), values = correlation
#     - C_max: maximum number of clusters to test
#     - k: number of nearest neighbors for local scaling
#
#     Returns:
#     - List of clusters (list of lists of features)
#     - Optimal number of clusters
#     - Alignment costs for each C
#     """
#     # Step 1: Extract features and index map
#     # Step 1: Extract all unique channels and index mapping
#     if included_series is None:
#         included_series = set()
#         for (s1, s2) in rank_correlations.keys():
#             included_series.add(s1)
#             included_series.add(s2)
#
#     channels = sorted(included_series)  # Ensure deterministic ordering
#     index_map = {feat: i for i, feat in enumerate(channels)}
#     n = len(channels)
#
#     # Step 2: Build distance matrix
#     distance_matrix = np.ones((n, n))
#     for (ch1, ch2), corr in rank_correlations.items():
#         if ch1 not in included_series or ch2 not in included_series:
#             continue
#         i, j = index_map[ch1], index_map[ch2]
#         dist = 1 - corr
#         distance_matrix[i, j] = distance_matrix[j, i] = dist
#
#     # Step 3: Compute local scales
#     sigma = compute_local_scales(distance_matrix, k)
#
#     # Step 4: Build affinity matrix
#     A = build_affinity_matrix(distance_matrix, sigma)
#
#     # Step 5: Normalize
#     L = normalize_affinity_matrix(A)
#
#     # Step 6: Eigen decomposition
#     eigvals, eigvecs = eigh(L)
#
#     # Step 7: For each C, compute cost
#     costs = []
#     for C in range(2, min(C_max + 1, n)):
#         X = eigvecs[:, :C]
#         X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
#         R = best_rotation(X_norm)
#         Z = X_norm @ R
#         cost = alignment_cost(Z)
#         costs.append(cost)
#     #############################################################
#     # Step 8: Choose best C (largest C with minimal cost)
#     min_cost = min(costs)
#     best_C = max(i + 2 for i, c in enumerate(costs) if c <= min_cost + 1e-4)
#
#     # Step 9: Final clustering
#     X_final = eigvecs[:, :best_C]
#     X_norm = X_final / np.linalg.norm(X_final, axis=1, keepdims=True)
#     R = best_rotation(X_norm)
#     Z = X_norm @ R
#     labels = assign_clusters(Z)
#
#     # Step 10: Group by clusters
#     clusters = defaultdict(list)
#     for i, label in enumerate(labels):
#         clusters[label].append(channels[i])
#
#     return list(clusters.values()), best_C, costs

