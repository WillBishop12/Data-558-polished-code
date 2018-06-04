"""Polished code release for kernel SVM"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.datasets import make_blobs
from multiprocessing.dummy import Pool
from timeit import default_timer as timer

#Number of CPUs for multiprocessing
cpu = 2

#------------------------------------------------------------------------
#Create simulated random data set with 4 classes, 4 features, and 100 observations (changeable by user)
X, y = make_blobs(n_samples=100, n_features=4, centers=4)

#Train-test split and standardize the data
X_train, X_test, y_train, y_test = train_test_split(X, y)

#Standardize X_train and X_test
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Standardize features according to training data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#------------------------------------------------------------------------
#Wrapper to calculate kernels, train one-vs-rest classifiers, and get multi-class predictions on training and test set
def multiclass_wrapper_ovr(X_train, y_train, X_test, y_test, kernel, X_val=None, **kwargs):
    """INPUTS:
    X_train = training features, y_train = training labels
    X_test = testing features, y_test = testing labels (optional)
    kernel = vectorized kernel funcion
    **kwargs = arguments of ovr_classifier function
    """
    K_train = kerneleval(kernel, X_train, X_train)
    K_test = kerneleval(kernel, X_train, X_test)
    if X_val is not None:
        K_val = kerneleval(kernel, X_train, X_val)
    else:
        K_val = None
    clf = ovr_classifier(K=K_train, y=y_train, K_val=K_val, **kwargs)
    train_pred = multiclass_from_ovr(K_train, clf)
    test_pred = multiclass_from_ovr(K_test, clf)
    train_error = 1 - np.mean(train_pred == y_train)
    if y_test is not None:
        test_error = 1 - np.mean(test_pred == y_test)
    else:
        test_error = 'N/A'
    visualize_training_ovr(clf=clf, K=K_train, y=y_train)
    return {
        'One-vs-rest classifiers': clf,
        'Training set predictions': train_pred,
        'Testing set predictions': test_pred,
        'Training error': train_error,
        'Testing error': test_error
    }

#Visualize training process for a multi-class one-vs-rest kernel SVM
def visualize_training_ovr(clf, K, y, costs=True, errors=True, classes=None):
    """
    INPUTS:
    clf = multiclass one-vs-rest classifier output by ovr_classifier
    K = kernel matrix of training features (only needed if errors = true)
    y = training labels (only needed if errors = true)
    costs = whether to graph cost function values
    errors = whether to graph training error
    classes = list classes to graph the one-vs-rest model for (one graph per class)
    """
    #If classes not specified, visualize models for all classes
    #Pseudocode
    if classes == None:
        classes = clf.keys()
    for c1 in classes:
        y = 2*(y == c1).astype(int) - 1
        x_axis = range(len(clf[c1]['Cost values']))
        labels = []
        if costs:
            plt.plot(x_axis, clf[c1]['Cost values'], label='Cost function')
            labels.append('Cost function')
        if errors:
            errors = [misclass(K, y, beta, classes=(-1, 1)) for beta in clf[c1]['Beta values']]
            plt.plot(x_axis, errors, label='Training error')
            labels.append('Training error')
        plt.title('Training process for class %d' %(c1))
        plt.xlabel('Iterations')
        plt.legend(labels)
        plt.show()


#------------------------------------------------------------------------
#Code to implement one-vs-rest classifiers
#Train one-vs-rest classifiers
def ovr_classifier(K, y, cross_val=True, folds=3, K_val=None, y_val=None,  
                   lamb_list=[2**-n for n in range(-3, 5)], **kwargs):
    """INPUTS:
    K = Gram matrix of training features
    y = training labels
    cross_val = whether to do cross-validation within training set
    - If True, use number of folds of the training set specified by "folds"
    - If False, use separate validation set specified by "K_val, y_val"
    lamb_list = list of lambda values to try
    **kwargs = arguments of kernel_svm function
    """
    #Iterate over all classes
    classes = np.unique(y)
    pool = Pool(cpu)
    #Define single_ovr_model within this function so it can take the appropriate parameters
    def single_ovr_model(c1):
        y_tr = (y == c1).astype(int)
        y_tr = 2*y_tr - 1
        if len(lamb_list) > 1 and cross_val == False:
            results_errors = {}
            y_val_ = 2*(y_val == c1).astype(int) - 1
            for lamb in lamb_list:
                lamb_results = kernel_svm(K=K, y=y_tr, lamb=lamb, **kwargs)
                lamb_beta = lamb_results['Beta values'][-1]
                results_errors[lamb] = {'result': lamb_results, 
                                     'error': misclass(K=K_val, y=y_val_, beta=lamb_beta, classes=(-1, 1))}
            lamb_opt = min(results_errors, key=lambda k: results_errors[k]['error'])
            result = results_errors[lamb_opt]['result']
        else:
            if len(lamb_list) > 1 and cross_val == True:
                lamb_opt = pick_optimal_lamb(lamb_list=lamb_list, K=K, y=y_tr, folds=folds, classes=(-1, 1), **kwargs)
            else:
                lamb_opt = lamb_list[0]
            result = kernel_svm(K=K, y=y_tr, lamb=lamb_opt, **kwargs)
        return (c1, {'Beta values': result['Beta values'], 'Cost values': result['Cost values']})
    results = pool.map(single_ovr_model, classes)
    pool.close()
    pool.join()
    return dict(results)

#Get multiclass predictions from one-vs-rest results
def multiclass_from_ovr(K, d):
    #Pick the class with the largest value of K.T.dot(beta) for each observation
    return np.array([max(d.keys(), key=lambda k: K.T.dot(d[k]['Beta values'][-1])[i]) for i in range(len(K.T))])

#Misclassification error from one-vs-rest results
def multiclass_error_ovr(X, y, classifiers):
    pred = multiclass_from_ovr(X, classifiers)
    return 1 - np.mean(pred == y)

#Pick optimal lambda value for fast gradient descent using k-folds cross validation
def pick_optimal_lamb(lamb_list, **kwargs):
    """
    Find optimal regularization parameter lambda for coordinate descent by k-folds cross validation.
    INPUTS:
    lamb_list = list of lambda values to check
    **kwargs = arguments of cv_error function
    """
    lamb_errors = {}
    for lamb in lamb_list:
        lamb_errors[lamb] = cv_error(lamb=lamb, **kwargs)
    return min(lamb_errors, key=lamb_errors.get)

#Cross-validation error
def cv_error(folds, K, y, classes=(-1, 1), **kwargs):
    """
    Calculates average misclassification error for a single instance of cross-validation.
    INPUTS:
    folds = number of folds to split training set
    K = Gram matrix of training features
    y = training labels
    classes = two classes represented in y
    **kwargs = arguments of kernel_svm
    """
    assert folds > 1
    error = []
    for f in range(folds):
        start = int(len(y) * f/folds)
        end = int(len(y) * (f+1)/folds)
        #out_idx = fold we leave out, in_idx = folds we keep in
        out_idx = np.r_[start:end]
        in_idx = np.r_[0:start, end:len(y)]
        y_out, y_in = y[out_idx], y[in_idx]
        K_in = K[in_idx][:, in_idx]
        #For validation, use k(x_i, x_j) where x_j is in left-out fold and x_i is not
        K_val = K[in_idx][:, out_idx]
        result = kernel_svm(K=K_in, y=y_in, **kwargs)
        beta = result['Beta values'][-1]
        error.append(misclass(K=K_val, y=y_out, beta=beta, classes=classes))
    return np.mean(error)

#Predict two classes from kernel matrix applied to training and tesing data
def predict(K, beta, classes=(-1, 1)):
    y_pred = np.sign(K.T.dot(beta))
    class_pred = classes[0]*(y_pred == -1) + classes[1]*(y_pred == 1)
    return class_pred

#Misclassification error
def misclass(K, y, beta, classes=(-1, 1)):
    """Computes misclassification error on output of coordinate descent.
    INPUTS: K = Gram matrix of training features, y = training labels,
    beta = weights on training observations
    classes = tuple where
        classes[0] = expected label in y when gradient descent predicts -1,
        classes[1] = expected label in y when gradient descent predicts +1.
    Assume cutoff of 0, i.e., when output of gradient descent < 0
    place in classes[0], otherwise in classes[1]"""
    
    #If y is a 2D array, convert it to 1D!
    if len(y.shape) > 1 and y.shape[1] == 1:
        y = y[:, 0]
    #If it's 2D with more than one column, throw error
    if len(y.shape) > 1 and y.shape[1] > 1:
        raise Exception("Wrong shape.")
    y_pred = predict(K, beta, classes)
    return 1 - np.mean(y_pred == y)

#----------------------------------------------------
#Setup for kernel SVM with smoothed hinge loss

#Functions for smoothed hinge loss and its gradient
def smoothed_hinge_loss(alpha, K, y, lamb, h=0.5):
    yt = y*K.dot(alpha)
    l_hh = (1+h-yt)**2 / (4*h) * (np.abs(1-yt) <= h) + (1-yt)*(yt < (1-h))
    return np.mean(l_hh) + lamb*np.dot(alpha, alpha)

def gradient_smoothed_hinge(alpha, K, y, lamb, h=0.5):
    yt = y*K.dot(alpha)
    l_hh_prime = -(1+h-yt)/(2*h)*y * (np.abs(1-yt) <= h) - y*(yt < (1-h))
    return np.mean(l_hh_prime[:, np.newaxis]*K, axis=0) + 2*lamb*alpha

#Test gradient calculation
def test_grad(eps=1e-6):
    n = 5
    for i in range(n):
        np.random.seed(0)
        beta = -np.random.normal(size=n)
        x = np.random.randn(n, n)
        K = sklearn.metrics.pairwise.rbf_kernel(x, x)
        y = np.random.choice([0, 1], size=5)
        lam = 0.5
        f1 = smoothed_hinge_loss(beta, K, y, lam)
        beta[i] = beta[i] + eps
        f2 = smoothed_hinge_loss(beta, K, y, lam)
        print('Estimated and calculated values of beta[', i, ']:', (f2-f1)/eps, gradient_smoothed_hinge(beta, K, y, lam)[i])
        assert np.isclose((f2-f1)/eps, gradient_smoothed_hinge(beta, K, y, lam)[i]), \
            'Estimated gradient ' + str((f2-f1)/eps) + ' is not approximately equal to the computed gradient ' \
            + str(gradient_smoothed_hinge(beta, K, y, lam)[i])
    print('Test passed')
  
#test_grad()

#Compute Gram matrix of one or two data sets given kernel function
def kerneleval(vectorized_kernel, X, Z=None):
    """
    INPUTS:
    X = data set
    Z = second data set (optional)
    vectorized_kernel = function that takes in two data sets and computes inner product between each pair of observations
    """
    if Z is None:
        K = vectorized_kernel(X, X)
    else:
        K = vectorized_kernel(X, Z)
    return K

#RBF kernel function
def rbf_kernel(X, Z, sigma=0.5):
    return np.exp(-1./(2*sigma**2)*((np.linalg.norm(X, axis=1)**2)[:, np.newaxis] \
                                    + (np.linalg.norm(Z, axis=1)**2)[np.newaxis, :] \
                                    - 2*np.dot(X, Z.T)))

def test_gram():  
    np.random.seed(0)
    X = np.random.randn(6, 5)
    Z = np.random.randn(7, 5)
    sigma = 10
    gram1 = rbf_kernel(X, Z, sigma)
    gram2 = sklearn.metrics.pairwise.rbf_kernel(X, Z, gamma=1./(2*sigma**2))
    assert np.allclose(gram1, gram2), 'Computed matrix' + str(gram1) + 'does not match that of scikit-learn:' + str(gram2)
    print('Test passed')
    
#test_gram()

#Polynomial kernel function
def poly_kernel(X, Z, b, p):
    return (X.dot(Z.T) + b)**p

#Polynomial kernel of degree 3 with b=1
def poly_kernel_3(X, Z):
    return poly_kernel(X, Z, b=1, p=3)

#Linear kernel function
def linear_kernel(X, Z):
    return X.dot(Z.T)

#Backtracking line search for step size
def backtracking(beta, K, y, lamb, eta_0=1, a=0.5, b=0.5, large_mag=False, max_iter=1000): 
    #Backtracking line search adapted from lab 2
    """
    Perform backtracking line search
    Inputs:
      - beta: Current point (i.e. coefficient estimate)
      - eta_0: Starting (maximum) step size
      - a: Constant used to define sufficient decrease condition
      - b: Fraction by which we decrease eta if the previous eta doesn't work
      - max_iter: Maximum number of iterations to run the algorithm
      - large_mag: Version for gradients of large magnitude overrides max_iter so eta can become small enough
        - Still hard-codes limit of 10^6 iterations
    Output:
      - eta: Step size to use
    """
    grad = gradient_smoothed_hinge(beta, K, y, lamb)
    norm_grad = np.linalg.norm(grad)
    found = False
    eta = eta_0
    i = 0    
    while found == False and (i < max_iter or (large_mag == True and i < 10**6)):
        if smoothed_hinge_loss(beta - eta*grad, K, y, lamb) < \
            smoothed_hinge_loss(beta, K, y, lamb) - a * eta * norm_grad**2:
            found = True
        elif (i == max_iter - 1 and large_mag == False):
            raise Exception('Maximum number of iterations of backtracking reached at step size %f' %(eta))
        elif i >= 10**6:
            raise Exception('Infinite loop warning at step size %f' %(eta))
            break
        else:
            eta *= b
            i += 1
    return eta

#Kernel SVM implementing fast gradient algorithm without maximum number of iterations
def kernel_svm(K, y, lamb, eta_0=50, eps=10**-3, large_mag=False, max_iter=None, **kwargs):    
    """INPUTS:
    eta_0 = initial step size for fast gradient algorithm
    K = Gram matrix of training features
    y = training labels
    lambda = regularization parameter
    eps = gradient size stopping criterion for fast gradient algorithm
    large_mag = whether to use large-magnitude version of kernel SVM
    max_iter = maximum number of iterations for fast gradient algorithm
    **kwargs = optional arguments of backtracking function (a and b parameters)
    """
    #Initialize variables beta, theta as zero vectors and compute initial gradient
    n = len(K)
    beta_0 = theta_0 = np.array([0] * n)
    #Keep track of two beta values for fast gradient rule
    beta = beta_0
    prev_beta = beta_0
    theta = theta_0
    #Gradient calculated based on theta instead of beta
    grad_theta = gradient_smoothed_hinge(theta, K, y, lamb)
    grad_beta = gradient_smoothed_hinge(beta, K, y, lamb)
    t = 0    
    #Keep track of cost function F(beta) so we can graph it in next step
    cost = smoothed_hinge_loss(beta, K, y, lamb)
    cost_values = [cost]    
    #Also keep track of beta itself so we can track misclassification error
    beta_values = [beta]    
    #Gradient descent calculation
    eta_old = eta_0
    while np.linalg.norm(grad_theta) > eps and (large_mag == False or t < max_iter):
        eta = backtracking(theta, K=K, y=y, lamb=lamb, eta_0=eta_old, large_mag=large_mag, **kwargs)        
        #Fast gradient rule
        prev_beta = beta
        beta = theta - eta * grad_theta
        theta = beta + t/(t + 3.) * (beta - prev_beta)        
        #Compute new gradient and cost
        grad_theta = gradient_smoothed_hinge(theta, K, y, lamb)
        grad_beta = gradient_smoothed_hinge(beta, K, y, lamb)
        cost = smoothed_hinge_loss(beta, K, y, lamb)
        cost_values.append(cost)
        beta_values.append(beta)
        eta_old = eta
        t += 1        
    return {'Cost values': cost_values, 'Beta values': beta_values}

#------------------------------------------------------------------------
#Run kernel SVM functions

#Linear kernel
start = timer()
result_linear = multiclass_wrapper_ovr(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
    cross_val=True, folds=3, kernel=linear_kernel, lamb_list=[.01, .1, 1, 10])
print("Validation score for linear kernel", 1 - result_linear['Testing error'], "\n")
end = timer()
print("Time for linear kernel ", end-start, "\n")

#Polynomial kernel of order 3
start = timer()
result_poly3 = multiclass_wrapper_ovr(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
    cross_val=True, folds=3, kernel=poly_kernel_3, lamb_list=[.01, .1, 1, 10])
print("Validation score for polynomial kernel of order 3", 1 - result_poly3['Testing error'], "\n")
end = timer()
print("Time for polynomial kernel of order 3 ", end-start, "\n")

#RBF kernel
start = timer()
result_rbf = multiclass_wrapper_ovr(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
    cross_val=True, folds=3, kernel=rbf_kernel, lamb_list=[.01, .1, 1, 10], eps=1e-6)
print("Validation score for RBF kernel", 1 - result_rbf['Testing error'], "\n")
end = timer()
print("Time for RBF kernel ", end-start, "\n")

#See which kernel had the best validation score, run it on the full training set, and predict on test data
best_score = min(model['Testing error'] for model in [result_rbf, result_poly3, result_linear])
start = timer()
if best_score == result_linear['Testing error']:
    print("Best kernel was linear. \n")
    final_kernel = multiclass_wrapper_ovr(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
        cross_val=True, folds=3, large_mag=True, max_iter=500,
        kernel=linear_kernel, lamb_list=[.01, .1, 1, 10])
elif best_score == result_poly3['Testing error']:
    print("Best kernel was polynomial. \n")
    final_kernel = multiclass_wrapper_ovr(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
        cross_val=True, folds=3, large_mag=True, max_iter=300, 
        kernel=poly_kernel_3, lamb_list=[.01, .1, 1, 10])
elif best_score == result_rbf['Testing error']:
    print("Best kernel was RBF. \n")
    final_kernel = multiclass_wrapper_ovr(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
        cross_val=True, folds=3, kernel=rbf_kernel, lamb_list=[.01, .1, 1, 10], eps=1e-6)
else:
    pass
end = timer()
print("Time for final kernel", end-start)
print("Validation score for final kernel", 1 - final_kernel['Testing error'])


#------------------------------------------------------------------------
#Scikit-learn version of kernel SVM

#Setup for one-vs rest kernel SVM
def sklearn_kernel_svm(X_train, y_train, X_test, y_test,
    cross_val=True, param_grid=None, **kwargs):
    """
    INPUTS:
    X_train, y_train = training features and labels
    X_test, y_test = testing features and labels
    cross_val = whether to wrap in GridSearchCV for cross-validation
    param_grid = parameter choices to cross-validate
    **kwargs = other arguments of sklearn.svm.SVC
    """
    svm_ovr = SVC(**kwargs)
    if cross_val:
        clf = GridSearchCV(estimator=svm_ovr, n_jobs=1, param_grid=param_grid)
    else:
        clf = svm_ovr
    start = timer()
    clf.fit(X_train, y_train)
    end = timer()
    runtime = end - start
    val_score = clf.score(X_test, y_test)
    return {'Validation score': val_score, 'Run time': runtime, 'Parameters': clf.best_params_}

kernel_param_grid = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [.01, .1, 1, 10, 100],
    }
kernel_svm_cv = sklearn_kernel_svm(X_train, y_train, X_test, y_test,
    param_grid=kernel_param_grid)
print("Kernel SVM with cross-validation (scikit-learn):", "Score", kernel_svm_cv['Validation score'],
    "Run time", kernel_svm_cv["Run time"], 'Best parameters', kernel_svm_cv['Parameters'], "\n")