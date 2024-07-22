import autograd.numpy as np
#import numpy as np
from autograd import grad
import time as time
from scipy.optimize import minimize

import demand.coefficients as coef

import gmm as gmm
import weightingmatrix as wm

def stage(ds, thetainit, W, avg_price_el, div_ratio):
    #----------------------------------------------------------------
    #   METHOD evaluateFC
    #----------------------------------------------------------------
     ## Compute the function and constraint values at x.

    def evaluateObjFct(x):
        return gmm.objfct(ds, x, ds.data, W, gmm.g, avg_price_el, div_ratio)

    def evaluateFC(x):
        start = time.time()
        print("Theta: " + str(x), flush=True)
        obj = evaluateObjFct(x)
        print("Objective function: " + str(obj), flush=True)
        print("Time: " + str(time.time() - start), flush=True)
        return obj

    #----------------------------------------------------------------
    #   METHOD evaluateGA
    #----------------------------------------------------------------

    objfct_grad = grad(evaluateObjFct)

    def evaluateGA(x):
        grad_objfct = objfct_grad(x)
        if np.any(np.isnan(grad_objfct)):
            eps = 1.0e-7
            obj = evaluateObjFct(x)
            for i in range(grad_objfct.shape[0]):
                add_to = np.zeros(x.shape[0])
                add_to[i] = eps
                grad_objfct[i] = (evaluateObjFct(x + add_to) - obj) / eps
            print("Evaluating gradient numerically...", flush=True)
        print("Gradient: " + str(grad_objfct), flush=True)
        return grad_objfct

    res = minimize(evaluateFC, thetainit, jac=evaluateGA, tol=1.0e-5)
    
    print(str(res), flush=True)
    
    return res.x



def est(ds, thetainit, W, K, avg_price_el, div_ratio):
    theta_1stage = stage(ds, thetainit, W, avg_price_el, div_ratio)
    print('First stage: ', flush=True)
    print(str(theta_1stage), flush=True)
    What = wm.W(ds, theta_1stage, K, avg_price_el, div_ratio)
    theta_2stage = stage(ds, theta_1stage, What, avg_price_el, div_ratio)
    print('Second stage: ', flush=True)
    print(str(theta_2stage), flush=True)
    return theta_2stage, What
