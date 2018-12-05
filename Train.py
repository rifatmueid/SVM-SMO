import numpy as np

from SMO_Solver import *

from Decision_functions import *

def train(SVMModel):
    num_checked = 0
    check_all = 1

    while (num_checked > 0) or (check_all):
        num_checked = 0
        if check_all:
            # loop over all the training set
            for i in range(SVMModel.alpha.shape[0]):
                check_res, SVMModel = check_error(i, SVMModel)
                num_checked += check_res
                if check_res:
                    obj_res = obj_function(SVMModel.alpha, SVMModel.label, SVMModel.kernel, SVMModel.feature_vec)
                    SVMModel._obj.append(obj_res)

        else:
            # loop over the cases where alpha aren't at their limits
            for i in np.where((SVMModel.alpha != 0) & (SVMModel.alpha != SVMModel.C))[0]:
                check_res, SVMModel = check_error(i, SVMModel)
                num_checked += check_res
                if check_res:
                    obj_res = obj_function(SVMModel.alpha, SVMModel.label, SVMModel.kernel, SVMModel.feature_vec)
                    SVMModel._obj.append(obj_res)

        if check_all == 1:
            check_all = 0
        elif num_checked == 0:
            check_all = 1

    return SVMModel

