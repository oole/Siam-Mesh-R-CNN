import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance
import numpy as np


def laplacian(W, normalized=True):
    """
    Returns the Laplacian for the given weight matrix

    """
    d = W.sum(axis=0)
    if not normalized:
        D = scipy.sparse.diags(d.adjecency_matrices.squeeze(), 0)
        L = D-W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype = W.dtype)
        L = I - D * W * D

    assert type(L) is scipy.sparse.csr.csr_matrix
    return L

def rescale_laplacian(L, lmax=2):
    """
    Rescale Laplacian eigenvalues to [-1,1].
    """
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L /= lmax / 2
    L -= I
    return L