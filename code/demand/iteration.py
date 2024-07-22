# %%
# Import packages
import autograd.numpy as np
#import numpy as np

# %%
# Iteration functions for BLP contraction mapping
# The following function come directly from PyBLP code (Conlon and Gortmaker, RAND, 2020) iteration.py, they are reproduced here so that the Autograd package is used rather than pure NumPy, citation in paper
def linf_norm(x):
    return np.max(np.abs(x))

def safe_norm(norm, x):
    with np.errstate(all='ignore'):
        value = norm(x)
    return value if np.isfinite(value) else 0

def squarem_iterator(initialguess, contraction, max_evaluations, norm, safe_norm, tol, scheme, step_min, step_max, step_factor):
    x = initialguess
    evaluations = 0
    while True:
        # first step
        x0, x = x, contraction(x)
        g0 = x - x0
        evaluations = evaluations + 1
        if evaluations >= max_evaluations or safe_norm(norm, g0) < tol:
            break
        
        # second step
        x1, x = x, contraction(x)
        g1 = x - x1
        evaluations = evaluations + 1
        if evaluations >= max_evaluations or safe_norm(norm, g1) < tol:
            break
            
        # compute step length
        r = g0
        v = g1 - g0
        if scheme == 1:
            alpha = (np.matmul(r.T, v)) / (np.matmul(v.T, v))
        elif scheme == 2:
            alpha = (np.matmul(r.T, r)) / (np.matmul(r.T, v))
        else:
            alpha = -np.sqrt((np.matmul(r.T, r)) / (np.matmul(v.T, v)))
            
        # bound the step length and update its bounds
        alpha = -np.maximum(step_min, np.minimum(step_max, -alpha))
        if -alpha == step_max:
            step_max *= step_factor
        if -alpha == step_min and step_min < 0:
            step_min *= step_factor
            
        # acceleration step
        with np.errstate(all='ignore'):
            x2, x = x, x0 - 2 * alpha * r + alpha**2 * v
            x3, x = x, contraction(x)
        
        # revert to the last evaluation if there were errors
        if not np.isfinite(x).all():
            x = x2
            continue
            
        # check for convergence
        evaluations = evaluations + 1
        if evaluations >= max_evaluations or safe_norm(norm, x - x3) < tol:
            break
            
    # determine whether there was convergence
    return x, evaluations < max_evaluations and np.isfinite(x).all()

def simple_iterator(initial, contraction, max_evaluations, norm, safe_norm, tol):
    x = initial
    evaluations = 0
    while True:
        x0, x = x, contraction(x)
        evaluations = evaluations + 1
        if evaluations >= max_evaluations or safe_norm(norm, x - x0) < tol:
            break
            
    # determine whether there was convergence
    return x, evaluations < max_evaluations and np.isfinite(x).all()
