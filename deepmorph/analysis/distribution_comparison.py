import numpy as np
from sklearn.neighbors import NearestNeighbors


def estimate_KL_div_by_knn(X, Y, k=1):
    '''Estimate the KL-divergence by the KNN method.
    Args:
        X: (n, d) sample drawn from distribution P.
        Y: (m, d) sample drawn from distribution Q.
        k: Number of the nearest neighbor.
    Return:
        Estimated D(P|Q)
    '''
    
    # Calculate the distances
    x_nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1).fit(X)
    y_nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1).fit(Y)
    
    dists_xx, _ = x_nn.kneighbors(X, k + 1, return_distance=True)
    dists_xy, _ = y_nn.kneighbors(X, k, return_distance=True)
    
    rho = dists_xx[:, -1]
    nu = dists_xy[:, -1]
    
    # Ignore the data points that encounter zero distances
    mask = (rho != 0) & (nu != 0)
    rho = rho[mask]
    nu = nu[mask]
    
    # Calculate the KL divergence
    n, m, d = len(rho), len(Y), len(X[0])

    kl_div =  d / n * np.sum(np.log(nu / rho)) + m / (n - 1)
    
    return kl_div

    
def estimate_J_div_by_knn(X, Y, k=1):
    '''Estimate the Jeffreys divergence by the KNN method'''
    return (estimate_KL_div_by_knn(X, Y, k=k) +  estimate_KL_div_by_knn(Y, X, k=k)) / 2


