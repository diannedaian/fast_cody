import numpy as np
from sklearn.cluster import KMeans

from .average_onto_simplex import average_onto_simplex


def skinning_clusters(W, D, T, k, l=2, num_clustering_features=10,
                      return_centroids=False, return_simplex_features=False):
    """ Skinning clusters.

    Parameters
    ----------
    W : numpy float array
        n x b skinning weights
    D : numpy float array
        b x 1 eigenvalue/weighing given to each skinning weight
    T : numpy int array
        T x 4 tet geometry
    k : int
        number of clusters
    l : int
        power to raise D to
    num_clustering_features : int
        number of features to use for clustering
    return_centroids : bool
        whether to return the centroids of the clusters
    return_simplex_features : bool
        whether to return the features averaged over each tet
    """
    num_clustering_features = min(num_clustering_features, W.shape[1])
    # need to average the skinning weights over each tet
    assert(T.shape[1] == 4, "only tets implemented so far for clustering")

    # Ensure D matches the number of columns in W (after orthonormalization, W might have fewer columns)
    # D should be reshaped to (b,) if it's (b, 1), and then sliced to match W
    D = D.flatten() if D.ndim > 1 else D
    num_modes_actual = W.shape[1]
    if len(D) > num_modes_actual:
        D = D[:num_modes_actual]
    elif len(D) < num_modes_actual:
        # If D has fewer elements than W has columns, pad with ones (shouldn't happen normally)
        D = np.pad(D, (0, num_modes_actual - len(D)), 'constant', constant_values=1.0)

    Wt = average_onto_simplex(W, T)
    # Wt2 = Wt / np.power(D, 2)
    Wt = Wt / np.power(D, l)
    Wt = Wt[:, 0:num_clustering_features]
    kmeans = KMeans(n_clusters=k, random_state=0).fit(Wt)
    l = kmeans.labels_

    if return_simplex_features:
        return l, Wt
    if return_centroids == True:
        return l, kmeans.cluster_centers_
    else:
        return l
