
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np 
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import itertools
from sklearn import preprocessing
from sklearn import linear_model
from scipy import optimize

# Functions # 
    
# Standardize columns 
def norm_func(arr):
    return (arr - np.mean(arr))/np.sqrt(np.var(arr))
   
# Get p's -- simplex values 
def get_design_lattice(k):
    
    center = np.repeat(1/k, k)
    ends = np.eye(k)
    
    mpt_combs = len(list(itertools.combinations(range(k), 2)))
    
    mid_arr = np.zeros((mpt_combs,k))
    verts = np.array(list(itertools.combinations(range(k), 2)))
    verts = np.vstack((np.repeat(np.arange(mpt_combs), 2), verts.ravel()))
    mid_arr[tuple(verts)] = .5
    
    return np.vstack((center, ends, mid_arr))

# Main function - takes data, simplex lattice coordinates, alpha regularization parm, num clusters // Returns tuple of potential optimum weights and error
def get_potential_weights(data, simplex, alpha = 3/16, num_clusters = 3, first = False, last_iter_mat = None):
    
    # Storing regression matrix from last iteration
    if last_iter_mat is not None:
        simplex_points_twoway_last = last_iter_mat[0]
        penalized_y = last_iter_mat[1]
        
    # Transform parameters for nelder-mead 
    def special_transform(lst):
        arr = np.array(lst)
        return np.exp(arr)/np.sum(np.exp(arr))
    
    # Function to be optimized by nelder-mead (quadratic canonical form)
    def nelder_func(x, betas):
        
        x = special_transform(x)
        
        x_combs = np.array(list(itertools.combinations(range(m), 2)))
        x_new = np.concatenate((x, np.array([np.prod(x[i]) for i in x_combs])))
        
        return np.dot(x_new, betas)
    
    # Iris dimensions 
    n = data.shape[0]
    m = data.shape[1]

    # Getting penalized y's based on lattice
    if first is True:
        penalized_y = []
    
    for i in range(simplex.shape[0]):
        weights = m*simplex[i]
        weighted_Z = np.matmul(data, np.diag(np.sqrt(weights)))
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, max_iter=1000, batch_size=1000).fit(X=weighted_Z)
        penalized_y.append(kmeans.inertia_/(n-1) + alpha*np.sum((weights-1)**2/(m-1)))
        
    # Creating design matrix for quadratic canonical model 
    poly = preprocessing.PolynomialFeatures(interaction_only=True,include_bias = False)
    simplex_points_twoway = poly.fit_transform(simplex)
    
    # Adding last design matrix if not first iteration
    if first is not True:
        simplex_points_twoway = np.vstack((simplex_points_twoway_last, simplex_points_twoway))
    
    # Fitting canonical quadratic based on simplex and penalized y values / Storing beta coefficients   
    reg = linear_model.LinearRegression(fit_intercept = False)
    
    reg.fit(simplex_points_twoway, np.array(penalized_y).reshape(-1,1))
    reg_mat = simplex_points_twoway, penalized_y
    
    betas = reg.coef_.T
        
    # Using BFGS to find potential optimum weights (where predicted penlized y is minimized based on coefficient estimated of quadratic canonical func)
    optim_output = optimize.minimize(nelder_func, list(np.zeros(m)), betas, method = "BFGS")
    # def get_trans_ub(simplex):
    #     return np.array([np.log(i[idx]*np.sum(np.exp(i))) for idx,i in enumerate(simplex[1:m+1])])
    
    # ub = get_trans_ub(simplex)
    # lb = np.repeat(None, m)
    # bounds = list(zip(lb, ub))
    # optim_output = optimize.minimize(nelder_func, list(np.zeros(m)), betas, bounds = bounds, method = "L-BFGS-B")
    
    # Getting potential optimum weights, using for weighted KMeans, and getting penalized y
    potential_opt = special_transform(optim_output.x)
    weights = m*potential_opt
    weighted_Z = np.matmul(data, np.diag(np.sqrt(weights)))
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X=weighted_Z)
    penalized_y_opt = kmeans.inertia_/(n-1) + alpha*np.sum((weights-1)**2/(m-1))
    
    # Comparing penlized y of potential optimum to penalized y estimated by quadratic 
    return potential_opt, np.abs(penalized_y_opt - optim_output.fun), reg_mat, penalized_y_opt


# Takes lattice, potential optimum point, and number of dimensions as arguments // Returns lattice coordinates of reduced simplex
def get_reduced_simplex(old_simplex, best_point, c_0, m):
    
    # Get delta weights based on potential optimum
    def get_delta(arr, ref):
        return np.sqrt(np.sum((ref - arr)**2))*c_0
    
    old_vertices = old_simplex[1:(m+1)]
    
    deltas = np.apply_along_axis(get_delta, 1, old_simplex[1:(1+m)], best_point)
    new_verts = np.array([deltas[i-1]*old_simplex[i] + (1 - deltas[i-1])*best_point for i in np.arange(1,m+1)])
    
    combs = np.array(list(itertools.combinations(range(m), 2)))
    twoway_combs = np.vstack([np.mean(new_verts[i], axis = 0) for i in combs])
    
    return np.vstack([best_point, new_verts, twoway_combs])

# Sequential exploration alg - Takes data, simplex lattice coords, epsilon marking when alg stops, alpha regularization term // Returns optimum and final error 
def sequential_exploration_alg(data, simplex, eps, alpha = 3/16, num_clusters = 3, max_iter = 100, show_results = False):
    
    c_0 = 1
    m = data.shape[1]
    
    for i in range(max_iter):
        
        if i == 0:
            results = get_potential_weights(data, simplex, alpha = alpha, num_clusters = num_clusters, first = True)
        else:
            results = get_potential_weights(data, reduced_simplex, alpha = alpha, num_clusters = num_clusters, last_iter_mat = results[2])

        if show_results:
            print(results[1])
        
        if results[1] < eps:
            return results
        
        c_0 = c_0*.5
        reduced_simplex = get_reduced_simplex(simplex, results[0], c_0, m)
        
        max_iter = max_iter - 1 
        
        if max_iter == 0:
            print("no optimum found")
            return results