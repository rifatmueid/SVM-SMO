# SVM implementation using SMO solver
# Done as a course project for ECE 570: Artificial Intelligence at IUPUI (fall 2018)

# To handle arrays and matrices 
import numpy as np

# To visualize data
import matplotlib.pyplot as plt

# Decision boundary contour
from matplotlib import path

#for reading mat file (feature vector)
from scipy.io import loadmat

from Kernels import *

from Decision_functions import *

from Train import *

class SMOModel:
    """SMOModel object container for initializations of the variable"""

    def __init__(self, feature_vec, label, C, kernel, alpha, bias, err):
        self.feature_vec = feature_vec  # training feature vector
        self.label = label  # class label vector
        self.C = C  # regularization or penalty parameter
        self.kernel = kernel  # kernel function: linear or rbf
        self.alpha = alpha  # lagrange multipliers
        self.bias = bias  # bias term in scalar
        self.err = err  # store error
        self._obj = []  # value of the objective function
        self.sz = len(self.feature_vec)  # training set size


# Set the tolerances
tolerance = 0.01  # error tolerance
epsilon = 0.01  # alpha tolerance

feature_vec_train = loadmat('Feature_S3.mat')
label = loadmat('Label_S3.mat')

feature_vec_train_scaled=feature_vec_train['feature1']
l1 = label['Label']
label = list(l1)
label = label[0]


# scaler = StandardScaler()
# feature_vec_train_scaled = scaler.fit_transform(feature_vec_train, label)
# label[label == 0] = -1

# Set SVM model parameters, initial values
C = 3000.0
sz = len(feature_vec_train_scaled)
initial_alphas = np.zeros(sz)
initial_bias = 0.0

# Get SVM model
SVMModel = SMOModel(feature_vec_train_scaled, label, C, rbf_kernel,
                 initial_alphas, initial_bias, np.zeros(sz))

# Initialize errors
initial_err = dec_function(SVMModel.alpha, SVMModel.label, SVMModel.kernel,
                                  SVMModel.feature_vec, SVMModel.feature_vec, SVMModel.bias) - SVMModel.label
SVMModel.err = initial_err

out = train(SVMModel)
fig, axes = plt.subplots()
grid, axes, poly = plot_boundary(out, axes)

# plt.show()
# plt.close()


# checking the output label
def check_label(x1, y1, x2, y2):
    shape = x1.shape
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    x2 = x2.reshape(-1)
    y2 = y2.reshape(-1)
    q = [(x1[i], y1[i]) for i in range(x1.shape[0])]
    p = path.Path([(x2[i], y2[i]) for i in range(x2.shape[0])])
    return p.contains_points(q).reshape(shape)


# total samples in output feature vector
total = len(out.feature_vec)

# check labels of the output feature vector
xx = np.array(out.feature_vec[:, 0])
yy = np.array(out.feature_vec[:, 1])
p1 = poly[:, 0]
p2 = poly[:, 1]
cc = check_label(xx, yy, p1, p2)

# initialization of the confusion matrix
conf_mat = np.zeros((3, 3), float)

# update the confusion matrix
for i in range(total):
    if label[i] == 1:
        if cc[i] == 1:
            conf_mat[0][0] += 1
        else:
            conf_mat[1][0] += 1
    else:
        if cc[i] == 0:
            conf_mat[1][1] += 1
        else:
            conf_mat[0][1] += 1

# confusion matrix in percentage
conf_mat = (conf_mat*100)/total
# overall accuracy
conf_mat[2][2] = conf_mat[0][0] + conf_mat[1][1]

# print(out.alpha)

print(conf_mat)
