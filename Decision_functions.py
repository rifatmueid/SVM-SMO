import numpy as np

from Kernels import *

# Objective function

def obj_function(alpha, feature_vec_target, kernel, feature_vec_train):
    """Returns objective function of the SVM based in the input SVMModel"""

    return np.sum(alpha) - 0.5 * np.sum(feature_vec_target * feature_vec_target * kernel(feature_vec_train, feature_vec_train) * alpha * alpha)


# Decision function

def dec_function(alpha, feature_vec_target, kernel, feature_vec_train, feature_vec_test, bias):
    """Implements the decision function of the SVM to the input feature vectors in `feature_vec_test`."""

    res = (alpha * feature_vec_target) @ kernel(feature_vec_train, feature_vec_test) - bias
    return res

