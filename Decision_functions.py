import numpy as np

import matplotlib.pyplot as plt


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


def plot_boundary(SVMModel, axes, resolution_grid=100, colors=('b', 'k', 'r')):
    """ SVM's decision boundary is plotted on the input axes object
        Decision boundary grid and axes object is returned"""

    # Generate coordinate grid of shape [resolution_grid features resolution_grid]
    # and evaluate the SVMModel over the entire space
    x_range = np.linspace(SVMModel.feature_vec[:, 0].min(), SVMModel.feature_vec[:, 0].max(), resolution_grid)
    y_range = np.linspace(SVMModel.feature_vec[:, 1].min(), SVMModel.feature_vec[:, 1].max(), resolution_grid)
    grid = [[dec_function(SVMModel.alpha, SVMModel.label,
                               SVMModel.kernel, SVMModel.feature_vec,
                               np.array([x_r, y_r]), SVMModel.bias) for y_r in y_range] for x_r in x_range]
    grid = np.array(grid).reshape(len(x_range), len(y_range))

    # Plot decision contours using grid and
    # make a scatter plot of training data
    axes.contour(x_range, y_range, grid, (-1, 0, 1), linewidths=(1, 0.5, 1),
               linestyles=('dashed', 'solid', 'dashed'), colors=colors)

    # co-ordinates of the contour
    poly_obj = plt.contour(x_range, y_range, grid, 0)
    poly = poly_obj.allsegs[0][0]

    axes.scatter(SVMModel.feature_vec[:, 0], SVMModel.feature_vec[:, 1],
               c=SVMModel.label, cmap=plt.cm.viridis, lw=0, alpha=0.5)

    # Plot the support vectors with alpha not equal zero
    mask = SVMModel.alpha != 0.0
    axes.scatter(SVMModel.feature_vec[:, 0][mask], SVMModel.feature_vec[:, 1][mask],
               c=SVMModel.label[mask], cmap=plt.cm.viridis)

    return grid, axes,  poly
