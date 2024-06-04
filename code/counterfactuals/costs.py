# %%
import numpy as np

import counterfactuals.priceequilibrium as pe
import counterfactuals.infrastructureequilibrium as ie
import counterfactuals.infrastructurefunctions as infr

import demand.blpextension as blp

# Function to recover per-user costs
def per_user_costs(theta, xis, ds, population, prices, cc_tot, stations, impute_MVNO=None, q_0=None, eps=0.001):
    # Calculate shares and Jacobian of shares with respect to prices
    shares = np.sum(blp.s_mj(ds, theta, ds.data, xis) * population[:,np.newaxis], axis=0)
    Jac = pe.s_jacobian_p(prices, cc_tot, ds, xis, theta, stations, population, impute_MVNO=impute_MVNO, q_0=q_0, eps=eps)
    
    # Determine per-user costs
    c_u = np.zeros(ds.J)
    for f, firm in enumerate(np.unique(ds.firms)):
        firm_cond = ds.firms == firm
        inv_Jac_firm = np.linalg.inv(Jac[np.ix_(firm_cond, firm_cond)])
        shares_firm = shares[firm_cond]
        c_u[firm_cond] = prices[firm_cond] + np.matmul(inv_Jac_firm, shares_firm)
        
    return c_u

# Function to recover per-base station costs
def per_base_station_costs(theta, xis, c_u, radius, bw_4g_equiv, lamda, ds, population, area, impute_MVNO=None, q_0=None, eps=0.001):
    MR = ie.pi_deriv_R(radius, bw_4g_equiv, lamda, ds, xis, theta, population, area, c_u, impute_MVNO=impute_MVNO, q_0=q_0, eps=eps)
    c_R = MR / infr.num_stations_deriv(radius, area[:,np.newaxis])
    return c_R
