import numpy as np

from Decision_functions import *
# Set the tolerances
tolerance = 0.01  # error tolerance
epsilon = 0.01  # alpha tolerance
C = 3000.0
initial_bias = 0.0

def update_model(j1, j2, SVMModel):
    # Skip this part if the chose alphas are the same
    if j1 == j2:
        return 0, SVMModel

    alpha1 = SVMModel.alpha[j1]
    alpha2 = SVMModel.alpha[j2]
    label1 = SVMModel.label[j1]
    label2 = SVMModel.label[j2]
    err1 = SVMModel.err[j1]
    err2 = SVMModel.err[j2]
    s1 = label1 * label2

    # Compute low bound, high bound and the bounds on new possible values of the alpha
    if label1 != label2:
        l_bound = max(0, alpha2 - alpha1)
        h_bound = min(SVMModel.C, SVMModel.C + alpha2 - alpha1)
    elif label1 == label2:
        l_bound = max(0, alpha1 + alpha2 - SVMModel.C)
        h_bound = min(SVMModel.C, alpha1 + alpha2)
    if l_bound == h_bound:
        return 0, SVMModel

    # Compute the kernels & the second derivative (sec_der)
    ker11 = SVMModel.kernel(SVMModel.feature_vec[j1], SVMModel.feature_vec[j1])
    ker12 = SVMModel.kernel(SVMModel.feature_vec[j1], SVMModel.feature_vec[j2])
    ker22 = SVMModel.kernel(SVMModel.feature_vec[j2], SVMModel.feature_vec[j2])
    sec_der = 2 * ker12 - ker11 - ker22

    # Compute new alpha 2 (alp2) if second derivative is negative
    if sec_der < 0:
        alp2 = alpha2 - label2 * (err1 - err2) / sec_der
        # Clip alp2 based on the bounds (l_bound & h_bound)
        if l_bound < alp2 < h_bound:
            alp2 = alp2
        elif (alp2 <= l_bound):
            alp2 = l_bound
        elif (alp2 >= h_bound):
            alp2 = h_bound

    # If second derivative is non-negative, move new alpha 2 (alp2) to the bound with the larger objective function value
    else:
        alpha_adj = SVMModel.alpha.copy()
        alpha_adj[j2] = l_bound
        # objective function output with alp2 = l_bound
        l_bound_obj = obj_function(alpha_adj, SVMModel.label, SVMModel.kernel, SVMModel.feature_vec)
        alpha_adj[j2] = h_bound
        # objective function output with alp2 = h_bound
        h_bound_obj = obj_function(alpha_adj, SVMModel.label, SVMModel.kernel, SVMModel.feature_vec)
        if l_bound_obj > (h_bound_obj + epsilon):
            alp2 = l_bound
        elif l_bound_obj < (h_bound_obj - epsilon):
            alp2 = h_bound
        else:
            alp2 = alpha2

    # Push alp2 to 0 or to C if very close to those
    if alp2 < 1e-8:
        alp2 = 0.0
    elif alp2 > (SVMModel.C - 1e-8):
        alp2 = SVMModel.C

    # If optimization is not achieved within epsilon, skip the following
    if np.abs(alp2 - alpha2) < epsilon * (alp2 + alpha2 + epsilon):
        return 0, SVMModel

    # Calculate the value of new alpha 1 (alp1)
    alp1 = alpha1 + s1 * (alpha2 - alp2)

    # Calculate the threshold for both cases
    bias1 = err1 + label1 * (alp1 - alpha1) * ker11 + label2 * (alp2 - alpha2) * ker12 + SVMModel.bias
    bias2 = err2 + label1 * (alp1 - alpha1) * ker12 + label2 * (alp2 - alpha2) * ker22 + SVMModel.bias

    # Set new updated threshold based on if they are bounded
    if 0 < alp1 and alp1 < C:
        bias_new = bias1

    elif 0 < alp2 and alp2 < C:
        bias_new = bias2

    # If both thresholds are bounded average them
    else:
        bias_new = (bias1 + bias2) * 0.5

    # Update SVMModel object with newly computed alpha
    SVMModel.alpha[j1] = alp1
    SVMModel.alpha[j2] = alp2

    # Update the errors
    # Errors for the optimized alpha is 0 if they are unbound
    for indx, alph in zip([j1, j2], [alp1, alp2]):
        if 0.0 < alph < SVMModel.C:
            SVMModel.err[indx] = 0.0

    # Calculate errors when the alpha is not optimized
    not_opt = [n for n in range(SVMModel.sz) if (n != j1 and n != j2)]
    SVMModel.err[not_opt] = SVMModel.err[not_opt] + \
                            label1 * (alp1 - alpha1) * SVMModel.kernel(SVMModel.feature_vec[j1], SVMModel.feature_vec[not_opt]) + \
                            label2 * (alp2 - alpha2) * SVMModel.kernel(SVMModel.feature_vec[j2], SVMModel.feature_vec[not_opt]) + SVMModel.bias - bias_new

    # Update the SVMModel threshold
    SVMModel.bias = bias_new

    return 1, SVMModel


def check_error(j2, SVMModel):
    label2 = SVMModel.label[j2]
    alpha2 = SVMModel.alpha[j2]
    err2 = SVMModel.err[j2]
    err_check = err2 * label2

    # Go forward if the error is within the tolerance (tolerance)
    if (err_check < -tolerance and alpha2 < SVMModel.C) or (err_check > tolerance and alpha2 > 0):

        if len(SVMModel.alpha[(SVMModel.alpha != 0) & (SVMModel.alpha != SVMModel.C)]) > 1:
            # Use the max difference in errors
            if SVMModel.err[j2] > 0:
                j1 = np.argmin(SVMModel.err)
            elif SVMModel.err[j2] <= 0:
                j1 = np.argmax(SVMModel.err)
            step_res, SVMModel = update_model(j1, j2, SVMModel)
            if step_res:
                return 1, SVMModel

        # Start at a random point and loop through the alphas which are non-zero or non-C
        for j1 in np.roll(np.where((SVMModel.alpha != 0) & (SVMModel.alpha != SVMModel.C))[0],
                          np.random.choice(np.arange(SVMModel.sz))):
            step_res, SVMModel = update_model(j1, j2, SVMModel)
            if step_res:
                return 1, SVMModel

        # Start at a random point and loop through aa the alphas
        for j1 in np.roll(np.arange(SVMModel.sz), np.random.choice(np.arange(SVMModel.sz))):
            step_res, SVMModel = update_model(j1, j2, SVMModel)
            if step_res:
                return 1, SVMModel

    return 0, SVMModel

