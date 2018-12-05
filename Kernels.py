import numpy as np

# kernels

def linear_kernel(features, label, bias=1):
    """Linear kernel with optional bias term ."""

    return features @ label.T + bias  # Note the @ operator for matrix multiplication


def rbf_kernel(features, label, gamma=3):
    """Calculation of the rbf kernel with free parameter gamma ."""

    if np.ndim(features) == 1 and np.ndim(label) == 1:
        res = np.exp(- np.linalg.norm(features - label) * gamma)
    elif (np.ndim(features) > 1 and np.ndim(label) == 1) or (np.ndim(features) == 1 and np.ndim(label) > 1):
        res = np.exp(- np.linalg.norm(features - label, axis=1) * gamma)
    elif np.ndim(features) > 1 and np.ndim(label) > 1:
        res = np.exp(- np.linalg.norm(features[:, np.newaxis] - label[np.newaxis, :], axis=2) * gamma)
    return res

