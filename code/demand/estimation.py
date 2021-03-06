import autograd.numpy as np
#import numpy as np
from autograd import grad
import time as time
from scipy.optimize import minimize

import demand.weightingmatrix as wm
import demand.coefficients as coef
import demand.gmm as gmm

def stage(ds, thetainit, W, avg_price_el, sigma):
    #----------------------------------------------------------------
    #   METHOD evaluateFC
    #----------------------------------------------------------------
     ## Compute the function and constraint values at x.

    def evaluateObjFct(x):
        return gmm.objfct(ds, x, ds.data, W, gmm.g, avg_price_el, sigma)

    def evaluateFC(x):
        start = time.time()
        print("Theta: " + str(x))
        obj = evaluateObjFct(x)
        print("Objective function: " + str(obj))
        print("Time: " + str(time.time() - start))
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
            print("Evaluating gradient numerically...")
        print("Gradient: " + str(grad_objfct))
        return grad_objfct

    res = minimize(evaluateFC, thetainit, jac=evaluateGA)
    
    print(str(res))
    
    return res.x



def est(ds, thetainit, W, K, avg_price_el, sigma):
    theta_1stage = stage(ds, thetainit, W, avg_price_el, sigma)
    print('First stage: ')
    print(str(theta_1stage))
    What = wm.W(ds, theta_1stage, K, avg_price_el, sigma)
    theta_2stage = stage(ds, theta_1stage, What, avg_price_el, sigma)
    print('Second stage: ')
    print(str(theta_2stage))
    return theta_2stage, What