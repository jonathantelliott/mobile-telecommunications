# %%
import copy

import numpy as np

from scipy.optimize import fsolve

import counterfactuals.infrastructurefunctions as infr
import counterfactuals.transmissionequilibrium as transeq
import counterfactuals.priceequilibrium as pe
import counterfactuals.welfare as welfare

import demand.blpextension as blp
import demand.dataexpressions as de

# %%
def pi_deriv_R(R, bw, gamma, ds, xis, theta, pop, market_size, c_u, symmetric=False, impute_MVNO={'impute': False}, q_0=None, eps=0.01):
    """
        Return the derivative of the operating income function with respect to cell radius, based on two-sided numerical derivative
    
    Parameters
    ----------
        R : ndarray
            (M,F) array of radii at which taking derivative
        bw : ndarray
            (M,F) array of bandwidth in MHz
        gamma : ndarray
            (M,) array of spectral efficiencies
        ds : DemandSystem
            contains all the data about our markets
        xis : ndarray 
            (M,J) matrix of vertical demand components
        theta : ndarray
            (K,) array of demand parameters
        pop : ndarray
            (M,) array of market populations
        market_size : ndarray
            (M,) array of geographic size of markets in km^2
        c_u : ndarray
            (J,) array of per-user costs
        symmetric : bool
            specifies whether the equilibrium solving for is symmetric (quicker to compute)
        impute_MVNO : dict
            dict with
            'impute' : bool (whether to impute the Qs for MVNO)
            'firms_share' (optional) : ndarray ((F-1,) array of whether firms share qualities with MVNOs)
            'include' (optional) : bool (whether to include MVNO Q in returned Q)
        q_0 : ndarray
            (M,F) array of initial guess of q
        eps : float
            size of perturbation to measure derivative

    Returns
    -------
        R_deriv : ndarray
            (M,F) array of firms' infrastructure FOCs for operating income
    """

    # Create high and low radius arrays
    R_high = R + eps
    R_low = R - eps

    # Calculate channel capacities at R (this will speed up later calculations)
    cc_R = np.zeros(R.shape)
    cc_high = np.zeros(R.shape)
    cc_low = np.zeros(R.shape)
    for m in range(R.shape[0]):
        for f in range(R.shape[1]):
            cc_R[m,f] = infr.rho_C_hex(bw[m,f], R[m,f], gamma[m])
            cc_high[m,f] = infr.rho_C_hex(bw[m,f], R_high[m,f], gamma[m])
            cc_low[m,f] = infr.rho_C_hex(bw[m,f], R_low[m,f], gamma[m])

    # Calculate number of stations with given radius
    num_stations_R = infr.num_stations(R, market_size[:,np.newaxis])
    num_stations_high = infr.num_stations(R_high, market_size[:,np.newaxis])
    num_stations_low = infr.num_stations(R_low, market_size[:,np.newaxis])

    # Create information about firms and markets
    firms = np.unique(ds.firms)
    M = R.shape[0]
    F = firms.shape[0]
    if impute_MVNO['impute']: # if we impute MVNO quality (o/w there are no MVNOs)
        firms = firms[:-1] # don't care about the MVNO firm in ds.firms
        F -= 1
        if impute_MVNO['include']: # if MVNO is needed for calculating shares
            F += 1
            
    # Expand variables if symmetric
    if symmetric:
        num_firms = firms.shape[0]
        cc_R = np.tile(cc_R, (1,num_firms))
        cc_high = np.tile(cc_high, (1,num_firms))
        cc_low = np.tile(cc_low, (1,num_firms))
        
        num_stations_R = np.tile(num_stations_R, (1,num_firms))
        num_stations_high = np.tile(num_stations_high, (1,num_firms))
        num_stations_low = np.tile(num_stations_low, (1,num_firms))
        
        xis = np.tile(xis, (1,num_firms))
        c_u = np.tile(c_u, (num_firms,))

    # Derivative for each firm
    R_deriv = np.zeros(R.shape)
    select_firms = np.ones(firms.shape[0], dtype=bool)
    if symmetric:
        select_firms[1:] = False
    for f, firm in enumerate(firms[select_firms]):
        # Create arrays for channel capacities with high and low R
        cc_high_f = np.copy(cc_R)
        cc_high_f[:,f] = cc_high[:,f]
        cc_low_f = np.copy(cc_R)
        cc_low_f[:,f] = cc_low[:,f]

        # Create arrays for number of stations
        stations_high = np.copy(num_stations_R)
        stations_high[:,f] = num_stations_high[:,f]
        stations_low = np.copy(num_stations_R)
        stations_low[:,f] = num_stations_low[:,f]

        # Calculate download speeds
        q_high = np.zeros((M,F))
        q_low = np.zeros((M,F))
        ds_temp = copy.deepcopy(ds)
        for m in range(M):
            select_m = np.arange(M) == m
            ds_temp.data = ds.data[select_m,:,:]
            q_high[m,:] = transeq.q(cc_high_f[select_m,:], ds_temp, xis[select_m,:], theta, stations_high[select_m,:], pop[select_m], impute_MVNO=impute_MVNO, q_0=q_0)[0,:] # 0 b/c we're doing this market-by-market
            q_low[m,:] = transeq.q(cc_low_f[select_m,:], ds_temp, xis[select_m,:], theta, stations_low[select_m,:], pop[select_m], impute_MVNO=impute_MVNO, q_0=q_0)[0,:] # 0 b/c we're doing this market-by-market

        # Update download speeds in characteristics
        ds_high = copy.deepcopy(ds)
        ds_low = copy.deepcopy(ds)
        qidx = ds.chars.index(ds.qname)
        firm_counts = np.unique(ds.firms, return_counts=True)[1]
        ds_high.data[:,:,qidx] = np.repeat(q_high, firm_counts, axis=1) # only works b/c products in order
        ds_low.data[:,:,qidx] = np.repeat(q_low, firm_counts, axis=1) # only works b/c products in order

        # Calculate demand for each product
        s_high = blp.s_mj(ds_high, theta, ds_high.data, xis) * pop[:,np.newaxis]
        s_low = blp.s_mj(ds_low, theta, ds_low.data, xis) * pop[:,np.newaxis]

        # Calculate profits
        pidx = ds.chars.index(ds.pname)
        pi_high = s_high * (ds.data[:,:,pidx] - c_u[np.newaxis,:])
        pi_low = s_low * (ds.data[:,:,pidx] - c_u[np.newaxis,:])

        # Sum up profits to firm level
        pi_high = np.sum(pi_high[:,ds.firms == firm], axis=1)
        pi_low = np.sum(pi_low[:,ds.firms == firm], axis=1)

        # Calculate derivative for fth radius
        R_deriv[:,f] = (pi_high - pi_low) / (2. * eps)

    # Return derivative
    return R_deriv

def R_foc(R, bw, gamma, ds, xis, theta, pop, market_size, c_u, c_R, symmetric=False, impute_MVNO={'impute': False}, q_0=None, eps=0.01):
    """
        Return the derivative of the overall profit function with respect to cell radius, based on two-sided numerical derivative
    
    Parameters
    ----------
        R : ndarray
            (M,F) array of radii at which taking derivative
        bw : ndarray
            (M,F) array of bandwidth in MHz
        gamma : ndarray
            (M,) array of spectral efficiencies
        ds : DemandSystem
            contains all the data about our markets
        xis : ndarray 
            (M,J) matrix of vertical demand components
        theta : ndarray
            (K,) array of demand parameters
        pop : ndarray
            (M,) array of market populations
        market_size : ndarray
            (M,) array of geographic size of markets in km^2
        c_u : ndarray
            (J,) array of per-user costs
        c_R : ndarray
            (M,F) array of base station fixed costs
        symmetric : bool
            specifies whether the equilibrium solving for is symmetric (quicker to compute)
        impute_MVNO : dict
            dict with
            'impute' : bool (whether to impute the Qs for MVNO)
            'firms_share' (optional) : ndarray ((F-1,) array of whether firms share qualities with MVNOs)
            'include' (optional) : bool (whether to include MVNO Q in returned Q)
        q_0 : ndarray
            (M,F) array of initial guess of q
        eps : float
            size of perturbation to measure derivative

    Returns
    -------
        foc : ndarray
            (M,F) array of firm-market infrastructure FOCs
    """

    # Solve for derivatives
    MR = pi_deriv_R(R, bw, gamma, ds, xis, theta, pop, market_size, c_u, symmetric=symmetric, impute_MVNO=impute_MVNO, q_0=q_0, eps=eps)
    stations_deriv = infr.num_stations_deriv(R, market_size[:,np.newaxis])

    # Solve for FOCs
    foc = MR - stations_deriv * c_R

    return foc

# def infrastructure_eqm(bw, gamma, ds, xis, theta, pop, market_size, c_u, c_R, R_0, p_0, symmetric=False, print_msg=False, impute_MVNO={'impute': False}, q_0=None, eps_R=0.01, eps_p=0.01, factor=100.):
#     """
#         Return the derivative of the profit function with respect to cell radius, based on two-sided numerical derivative
    
#     Parameters
#     ----------
#         bw : ndarray
#             (M,F) or (M,1) array of bandwidth in MHz
#         gamma : ndarray
#             (M,) array of spectral efficiencies
#         ds : DemandSystem
#             contains all the data about our markets
#         xis : ndarray 
#             (M,J*F) or (M,J) matrix of vertical demand components
#         theta : ndarray
#             (K,) array of demand parameters
#         pop : ndarray
#             (M,) array of market populations
#         market_size : ndarray
#             (M,) array of geographic size of markets in km^2
#         c_u : ndarray
#             (J*F,) or (J,) array of per-user costs
#         c_R : ndarray
#             (M,F) or (M,) array of per-tower costs
#         R_0 : ndarray
#             (M,F) or (M,1) array of initial guess of firm-market radii
#         p_0 : ndarray
#             (J*F,) or (J,) array of initial guess of prices
#         symmetric : bool
#             specifies whether the equilibrium solving for is symmetric (quicker to compute)
#         print_msg : bool
#             determines whether or not to print inputs and output of root solver
#         impute_MVNO : dict
#             dict with
#             'impute' : bool (whether to impute the Qs for MVNO)
#             'firms_share' (optional) : ndarray ((F-1,) array of whether firms share qualities with MVNOs)
#             'include' (optional) : bool (whether to include MVNO Q in returned Q)
#         q_0 : ndarray
#             (M,F) array of initial guess of q
#         eps_R : float
#             size of perturbation to measure radius derivative
#         eps_p : float
#             size of perturbation to measure price derivative
#         factor : float
#             size of the factor for fsolve, must be in interval [0.1, 100]

#     Returns
#     -------
#         R_star : ndarray
#             (M,F) array of firms' optimal infrastrucuture choice
#         p_star : ndarray
#             (J,) array of firms' optimal prices
#         q_star : ndarray
#             (M,F) array of qualities that result from prices and infrastructure
#     """

#     # Determine sizes of infrastructure and price arrays
#     R_shape = (ds.data.shape[0],np.unique(ds.firms).shape[0])
#     p_shape = (ds.data.shape[1],)
    
#     # Save parameters useful later
#     F = np.unique(ds.firms).shape[0]
#     pidx = ds.chars.index(ds.pname)
        
#     # Solve for the equilibrium using FOCs in two steps
#     ctr = 0
#     while True:
#         # Print current guess of parameters
#         if print_msg:
#             print(f"Iteration {ctr}")
#             print(f"R_{ctr}: {R_0}")
#             print(f"p_{ctr}: {p_0}")
        
#         # Reshape R_0
#         R_0_reshape = np.reshape(R_0, (R_shape[0],1)) if symmetric else np.reshape(R_0, R_shape)
        
#         # Solve for the channel capacity implied by radius R - NOTE: parallelize this for large number of markets
#         cc = np.zeros(R_0_reshape.shape)
#         for m in range(R_0_reshape.shape[0]):
#             for f in range(R_0_reshape.shape[1]):
#                 cc[m,f] = infr.rho_C_hex(bw[m,f], R_0_reshape[m,f], gamma[m])

#         # Solve for the number of stations implied by radius R
#         stations = infr.num_stations(R_0_reshape, market_size)
        
#         # Update ds
#         ds.data[:,:,pidx] = np.tile(p_0[np.newaxis,:], (1,F if symmetric else 1))
        
# #         # Solve for optimal prices given infrastructure
# #         def foc_p(p, p_other, position, return_all=False):
# #             """Returns the FOC with respect to price."""
# #             price = np.concatenate((p_other[:position], p, p_other[position:]))
# #             ds.data[:,:,pidx] = np.tile(price[np.newaxis,:], (1,F if symmetric else 1))
# #             foc = pe.p_foc(price, c_u, cc, ds, xis, theta, stations, pop, symmetric=symmetric, impute_MVNO=impute_MVNO, q_0=q_0, eps=eps_p)
# #             #print(f"p: {p}, foc: {foc}")
# #             if return_all:
# #                 return foc
# #             else:
# #                 return foc[np.arange(price.shape[0]) == position]
# #         ctr_p = 0
# #         print("FOC matrix")
# #         p_linspace = np.linspace(0., 50., 20)
# #         min_foc = 10000000000.
# #         argmin_foc = np.array([-99., -99,])
# #         for i in range(20):
# #             for j in range(20):
# #                 foc_pp = foc_p(np.array([p_linspace[i]]), np.array([p_linspace[j]]), 0, return_all=True)
# #                 print(f"p_1 = {np.round(p_linspace[i], 2)}, p_2 = {np.round(p_linspace[j], 2)}: {np.round(foc_pp, 3)}")
# #                 if np.max(np.abs(foc_pp)) < min_foc:
# #                     min_foc = np.max(np.abs(foc_pp))
# #                     argmin_foc = np.array([p_linspace[i], p_linspace[j]])
# #         print(f"min: {min_foc}")
# #         print(f"argmin: {argmin_foc}")

# #         while True:
# #             for i in range(p_0.shape[0]):
# #                 if i < 2:
# #                     print(f"i: {i}")
# #                     import matplotlib.pyplot as plt
# #                     p_linspace = np.linspace(0., 100., 50)
# #                     foc_linspace = np.zeros(p_linspace.shape)
# #                     for j in range(50):
# #                         foc_linspace[j] = foc_p(np.array([p_linspace[j]]), p_0[np.arange(p_0.shape[0]) != i], i)[0]
# #                     plt.plot(p_linspace, foc_linspace)
# #                     plt.show()
# #                     print(foc_linspace)
# #                 p_i_star, infodict, ier, msg = fsolve(foc_p, np.array([p_0[i]]), args=(p_0[np.arange(p_0.shape[0]) != i], i), full_output=True, factor=factor, xtol=5.0e-07)
# #                 # Update p* if there are no issues with the solver
# #                 if ier != 1:
# #                     print(f"Price equilibrium computation failed for following reason: {msg}. Additional information: {infodict}")
# #                     p_0 = p_0 * np.nan
# #                     break
# #                 else:
# #                     p_0[i] = p_i_star[0]
# #             if np.all(foc_p(np.array([]), p_0, 0, return_all=True) < 0.00001): # all the FOCs are satisfied
# #                 break
# #             ctr_p = ctr_p + 1
# #             if ctr_p > 1000:
# #                 print(f"Unable to find price equilibrium. Solver reached maximum iterations.")
# #                 break
# #         if print_msg:
# #             print(f"p*_{ctr}: {p_0}")
#         # Solve for optimal prices given infrastructure
#         def foc_p(p):
#             """Returns the FOC with respect to price."""
#             ds.data[:,:,pidx] = np.tile(p[np.newaxis,:], (1,F if symmetric else 1))
#             foc = pe.p_foc(p, c_u, cc, ds, xis, theta, stations, pop, symmetric=symmetric, impute_MVNO=impute_MVNO, q_0=q_0, eps=eps_p)
#             #print(f"p: {p}, foc: {foc}")
#             return foc
#         p_1, infodict, ier, msg = fsolve(foc_p, p_0, full_output=True, factor=factor, xtol=5.0e-07)
        
#         # Update p* if there are no issues with the solver
#         if ier != 1:
#             raise ValueError(f"Price equilibrium computation failed for following reason: {msg}. Additional information: {infodict}")
#         else:
#             p_0 = p_1
#         if print_msg:
#             print(f"p*_{ctr}: {p_0}")
            
#         # Update ds with optimal price
#         ds.data[:,:,pidx] = np.tile(p_0[np.newaxis,:], (1,F if symmetric else 1))

#         # Solve for the infrastructure FOCs
#         foc_infr = lambda R: np.reshape(R_foc(np.reshape(R, (R_shape[0],1)) if symmetric else np.reshape(R, R_shape), bw, gamma, ds, xis, theta, pop, market_size, c_u, c_R, symmetric=symmetric, impute_MVNO=impute_MVNO, q_0=q_0, eps=eps_R), (-1,))
        
#         # Determine if infrastructure is optimal
#         R_0_foc_infr = foc_infr(R_0)
#         if print_msg:
#             print(f"foc_R(R_{ctr}, p*_{ctr}): {R_0_foc_infr}")
#         if np.all(np.abs(R_0_foc_infr) < 0.001):
#             print("")
#             break
            
#         # Determine optimal infrastructure
#         R_1, infodict, ier, msg = fsolve(foc_infr, R_0, full_output=True, factor=factor)
        
#         # Update R* if there are no issues with the solver
#         if ier != 1:
#             print(f"Infrastructure equilibrium computation failed for following reason: {msg}. Additional information: {infodict}")
#             R_0 = R_1 * np.nan
#             break
#         else:
#             R_0 = R_1
#         if print_msg:
#             print(f"R*_{ctr}: {R_0}\n")
        
#         # Update counter
#         ctr = ctr + 1
#         if ctr > 1000:
#             p_0 = p_1 * np.nan
#             R_0 = R_1 * np.nan
#             print(f"Unable to find equilibrium. Solver reached maximum evaluations.")
            
#     if np.any(np.isnan(p_0)) or np.any(np.isnan(R_0)):
#         print(f"Did not find equilibrium.")
#         return p_0 * np.nan, R_0 * np.nan, np.ones(R_shape) * np.nan
            
#     # Save p* and R*
#     p_star = np.copy(p_0)
#     R_star = np.reshape(np.copy(R_0), (R_shape[0],1)) if symmetric else np.reshape(np.copy(R_0), R_shape)
#     if symmetric:
#         R_star = np.tile(R_star, (1,R_shape[1]))
#         p_star = np.tile(p_star, (R_shape[1],))
#         bw = np.tile(bw, (1,R_shape[1]))
#         xis = np.tile(xis, (1,R_shape[1]))

#     # Calculate implied channel capacities
#     cc = np.zeros(R_shape)
#     for m in range(R_shape[0]):
#         for f in range(R_shape[1]):
#             cc[m,f] = infr.rho_C_hex(bw[m,f], R_star[m,f], gamma[m])

#     # Calculate implied stations
#     stations = infr.num_stations(R_star, market_size)

#     # Calculate implied download speeds
#     q_star = np.zeros(R_shape)
#     M = R_shape[0]
#     ds_temp = copy.deepcopy(ds)
#     pidx = pidx = ds.chars.index(ds.pname)
#     ds_temp.data[:,:,pidx] = p_star
#     for m in range(M):
#             select_m = np.arange(M) == m
#             ds_temp.data = ds.data[select_m,:,:]
#             q_star[m,:] = transeq.q(cc[select_m,:], ds_temp, xis[select_m,:], theta, stations[select_m,:], pop[select_m], impute_MVNO=impute_MVNO, q_0=q_0)[0,:] # 0 b/c we're doing this market-by-market
    
#     # Add MVNOs if imputing MVNO
#     if impute_MVNO['impute']:
#         if impute_MVNO['include']:
#             q_star = np.concatenate((q_star, transeq.q_MVNO(q_star, impute_MVNO['firms_share'])[:,np.newaxis]), axis=1)

#     return R_star, p_star, q_star

def combine_focs(R, p, bw, gamma, ds, xis, theta, pop, market_size, c_u, c_R, symmetric=False, print_msg=False, impute_MVNO={'impute': False}, q_0=None, eps_R=0.01, eps_p=0.01):
    """
        Return a combined array of FOCs that characterize an equilibrium, based on two-sided numerical derivative
    
    Parameters
    ----------
        R : ndarray
            (M,F) array of firm-market radii
        p : ndarray
            (J,) array of prices
        bw : ndarray
            (M,F) array of bandwidth in MHz
        gamma : ndarray
            (M,) array of spectral efficiencies
        ds : DemandSystem
            contains all the data about our markets
        xis : ndarray 
            (M,J) matrix of vertical demand components
        theta : ndarray
            (K,) array of demand parameters
        pop : ndarray
            (M,) array of market populations
        market_size : ndarray
            (M,) array of geographic size of markets in km^2
        c_u : ndarray
            (J,) array of per-user costs
        c_R : ndarray
            (M,F) array of per-tower costs
        symmetric : bool
            specifies whether the equilibrium solving for is symmetric (quicker to compute)
        print_msg : bool
            determines whether or not to print inputs and output
        impute_MVNO : dict
            dict with
            'impute' : bool (whether to impute the Qs for MVNO)
            'firms_share' (optional) : ndarray ((F-1,) array of whether firms share qualities with MVNOs)
            'include' (optional) : bool (whether to include MVNO Q in returned Q)
        q_0 : ndarray
            (M,F) array of initial guess of q
        eps_R : float
            size of perturbation to measure radius derivative
        eps_p : float
            size of perturbation to measure price derivative
    Returns
    -------
        foc : ndarray
            (M*F + J,) flattened array of FOCs (infrastructure then price)
    """

    if print_msg:
        print(f"R: {R}")
        print(f"p: {p}")
        
    F = np.unique(ds.firms).shape[0]
        
    # Update price
    pidx = ds.chars.index(ds.pname)
    ds.data[:,:,pidx] = np.tile(p[np.newaxis,:], (1,F if symmetric else 1))

    # Solve for the infrastructure FOCs
    infr_FOCs = R_foc(R, bw, gamma, ds, xis, theta, pop, market_size, c_u, c_R, symmetric=symmetric, impute_MVNO=impute_MVNO, q_0=q_0, eps=eps_R)

    # Solve for the channel capacity implied by radius R - NOTE: parallelize this for large number of markets
    cc = np.zeros(R.shape)
    for m in range(R.shape[0]):
        for f in range(R.shape[1]):
            cc[m,f] = infr.rho_C_hex(bw[m,f], R[m,f], gamma[m])

    # Solve for the number of stations implied by radius R
    stations = infr.num_stations(R, market_size)
    
    # Create information about firms and markets used for quality
    firms = np.unique(ds.firms)
    M = R.shape[0]
    F = firms.shape[0] # defined before but can change depending on what we're doing with MVNOs
    if impute_MVNO['impute']: # if we impute MVNO quality (o/w there are no MVNOs)
        firms = firms[:-1] # don't care about the MVNO firm in ds.firms
        F -= 1
        if impute_MVNO['include']: # if MVNO is needed for calculating shares
            F += 1
    cc_expand = np.copy(cc)
    stations_expand = np.copy(stations)
    xis_expand = np.copy(xis)
    if symmetric: # expand variables if symmetric
        num_firms = firms.shape[0]
        cc_expand = np.tile(cc_expand, (1,num_firms))
        stations_expand = np.tile(stations_expand, (1,num_firms))
        xis_expand = np.tile(xis_expand, (1,num_firms))
    
    # Solve for download speeds
    q = np.zeros((M,F))
    ds_temp = copy.deepcopy(ds)
    for m in range(M):
        select_m = np.arange(M) == m
        ds_temp.data = ds.data[select_m,:,:]
        q[m,:] = transeq.q(cc_expand[select_m,:], ds_temp, xis_expand[select_m,:], theta, stations_expand[select_m,:], pop[select_m], impute_MVNO=impute_MVNO, q_0=q_0)[0,:] # 0 b/c we're doing this market-by-market
    
    # Update download speed
    qidx = ds.chars.index(ds.qname)
    firm_counts = np.unique(ds.firms, return_counts=True)[1]
    ds.data[:,:,qidx] = np.repeat(q, firm_counts, axis=1) # works b/c products are in order

    # Solve for the pricing FOCs
    price_FOCs = pe.p_foc(p, c_u, cc, ds, xis, theta, stations, pop, symmetric=symmetric, impute_MVNO=impute_MVNO, q_0=q_0, eps=eps_p)

    # Combine FOCs into flattened array
    foc = np.concatenate((np.reshape(infr_FOCs, (-1,)), price_FOCs))

    if print_msg:
        #qs = transeq.q(cc, ds, xis, theta, stations, pop, impute_MVNO=impute_MVNO, q_0=q_0)
#         print(f"Ex: {de.E_x(ds, theta, ds.data, np.tile(qs, (R.shape[1])), ds.data[:,:,ds.chars.index(ds.dlimname)], blp.ycX(ds, theta, ds.data))[0,:,:]}")
        print(f"s_j: {np.mean(blp.s_mj(ds, theta, ds.data, np.tile(xis, (1,F)) if symmetric else xis), axis=0)}")
        #print(f"q: {np.mean(qs, axis=0)}")
        #print(f"E[x*]: {np.mean(de.E_x(ds, theta, ds.data, np.tile(qs, (R.shape[1])), ds.data[:,:,ds.chars.index(ds.dlimname)], blp.ycX(ds, theta, ds.data)), axis=0)}")
        #print(f"E[u(x*)]: {np.mean(de.E_u(ds, theta, ds.data, np.tile(qs, (R.shape[1])), ds.data[:,:,ds.chars.index(ds.dlimname)], blp.ycX(ds, theta, ds.data)), axis=0)}")
        print(f"foc: {foc}")

    return foc

def reshape_inputs(foc_shape, R_shape, p_shape, symmetric=False):
    """
        Return reshaped array of FOCs
    
    Parameters
    ----------
        foc_shape : ndarray
            (M*F + J,) flattened array of FOCs (infrastructure then price)
        R_shape : tuple
            size of infrastructure array
        p_shape : tuple
            size of price array
        symmetric : bool
            specifies whether the equilibrium solving for is symmetric (quicker to compute)
    Returns
    -------
        R : ndarray
            (M,F) array of infrastructure
        p : ndarray
            (J,) array of prices
    """
    
    if symmetric:
        R = np.reshape(foc_shape[:R_shape[0]], (R_shape[0],1))
        p = foc_shape[R_shape[0]:]
    else:
        R = np.reshape(foc_shape[:np.prod(R_shape)], R_shape)
        p = foc_shape[np.prod(R_shape):]

    return R, p

def infrastructure_eqm(bw, gamma, ds, xis, theta, pop, market_size, c_u, c_R, R_0, p_0, symmetric=False, print_msg=False, impute_MVNO={'impute': False}, q_0=None, eps_R=0.01, eps_p=0.01, factor=100., R_fixed=False):
    """
        Return the derivative of the profit function with respect to cell radius, based on two-sided numerical derivative
    
    Parameters
    ----------
        bw : ndarray
            (M,F) or (M,) array of bandwidth in MHz
        gamma : ndarray
            (M,) array of spectral efficiencies
        ds : DemandSystem
            contains all the data about our markets
        xis : ndarray 
            (M,J*F) or (M,J) matrix of vertical demand components
        theta : ndarray
            (K,) array of demand parameters
        pop : ndarray
            (M,) array of market populations
        market_size : ndarray
            (M,) array of geographic size of markets in km^2
        c_u : ndarray
            (J*F,) or (J,) array of per-user costs
        c_R : ndarray
            (M,F) or (M,) array of per-tower costs
        R_0 : ndarray
            (M,F) or (M,1) array of initial guess of firm-market radii
        p_0 : ndarray
            (J*F,) or (J,) array of initial guess of prices
        symmetric : bool
            specifies whether the equilibrium solving for is symmetric (quicker to compute)
        print_msg : bool
            determines whether or not to print inputs and output of root solver
        impute_MVNO : dict
            dict with
            'impute' : bool (whether to impute the Qs for MVNO)
            'firms_share' (optional) : ndarray ((F-1,) array of whether firms share qualities with MVNOs)
            'include' (optional) : bool (whether to include MVNO Q in returned Q)
        q_0 : ndarray
            (M,F) array of initial guess of q
        eps_R : float
            size of perturbation to measure radius derivative
        eps_p : float
            size of perturbation to measure price derivative
        factor : float
            size of the factor for fsolve, must be in interval [0.1, 100]
        R_fixed : bool
            determine whether to allow R to respond in the equilibrium, if True use R_0 as R*
    Returns
    -------
        R_star : ndarray
            (M,F) array of firms' optimal infrastrucuture choice
        p_star : ndarray
            (J,) array of firms' optimal prices
        q_star : ndarray
            (M,F) array of qualities that result from prices and infrastructure
        successful : bool
            True if all equilibrium computations successful, False otherwise
    """

    # Determine sizes of infrastructure and price arrays
    R_shape = (ds.data.shape[0],np.unique(ds.firms).shape[0])
    p_shape = (ds.data.shape[1],)

    # Define FOC 
    def eqm_foc(x):
        x_shape = x.shape # save shape of x (b/c need it later and going to transform x)
        if R_fixed: # include R_0 for R if R is fixed
            x = np.concatenate((np.reshape(R_0, (-1,)), x))
        foc = combine_focs(reshape_inputs(x, R_shape, p_shape, symmetric=symmetric)[0], reshape_inputs(x, R_shape, p_shape, symmetric=symmetric)[1], bw, gamma, ds, xis, theta, pop, market_size, c_u, c_R, symmetric, print_msg=print_msg, impute_MVNO=impute_MVNO, q_0=q_0, eps_p=eps_p, eps_R=eps_R)
        if R_fixed: # only take price FOCs if R is fixed
            foc = foc[-x_shape[0]:]
        return foc

    # Solve for the equilibrium
    Rp_star, infodict, ier, msg = fsolve(eqm_foc, p_0 if R_fixed else np.concatenate((np.reshape(R_0, (-1,)), p_0)), full_output=True, factor=factor)
    if R_fixed: # include R_0 for R if R is fixed
        Rp_star = np.concatenate((np.reshape(R_0, (-1,)), Rp_star))
    R_star, p_star = reshape_inputs(Rp_star, R_shape, p_shape, symmetric=symmetric)
    if symmetric:
        R_star = np.tile(R_star, (1,R_shape[1]))
        p_star = np.tile(p_star, (R_shape[1],))
        bw = np.tile(bw, (1,R_shape[1]))
        xis = np.tile(xis, (1,R_shape[1]))
    
    # Print error message if failed to converge
    if ier != 1:
        print(f"Equilibrium computation failed for following reason: {msg}. Additional information: {infodict}")

    # Calculate implied channel capacities
    cc = np.zeros(R_shape)
    for m in range(R_shape[0]):
        for f in range(R_shape[1]):
            cc[m,f] = infr.rho_C_hex(bw[m,f], R_star[m,f], gamma[m])

    # Calculate implied stations
    stations = infr.num_stations(R_star, market_size)

    # Calculate implied download speeds
    q_star = np.zeros(R_shape)
    M = R_shape[0]
    ds_temp = copy.deepcopy(ds)
    pidx = ds.chars.index(ds.pname)
    ds_temp.data[:,:,pidx] = p_star
    for m in range(M):
        select_m = np.arange(M) == m
        ds_temp.data = ds.data[select_m,:,:]
        q_star[m,:] = transeq.q(cc[select_m,:], ds_temp, xis[select_m,:], theta, stations[select_m,:], pop[select_m], impute_MVNO=impute_MVNO, q_0=q_0)[0,:] # 0 b/c we're doing this market-by-market
    
    # Add MVNOs if imputing MVNO
    if impute_MVNO['impute']:
        if impute_MVNO['include']:
            q_star = np.concatenate((q_star, transeq.q_MVNO(q_star, impute_MVNO['firms_share'])[:,np.newaxis]), axis=1)

    success = ier == 1
    return R_star, p_star, q_star, success

def bw_foc(bw, gamma, ds, xis, theta, pop, market_size, c_u, c_R, R_0, p_0, symmetric=False, print_msg=False, impute_MVNO={'impute': False}, q_0=None, eps_R=0.01, eps_p=0.01, eps_bw=0.01, factor=100., include_logit_shock=True, adjust_c_R=False):
    """
        Return the derivative of the profit function with respect to cell radius, based on two-sided numerical derivative
    
    Parameters
    ----------
        bw : ndarray
            (M,F) or (M,) array of bandwidth in MHz
        gamma : ndarray
            (M,) array of spectral efficiencies
        ds : DemandSystem
            contains all the data about our markets
        xis : ndarray 
            (M,J*F) or (M,J) matrix of vertical demand components
        theta : ndarray
            (K,) array of demand parameters
        pop : ndarray
            (M,) array of market populations
        market_size : ndarray
            (M,) array of geographic size of markets in km^2
        c_u : ndarray
            (J*F,) or (J,) array of per-user costs
        c_R : ndarray
            (M,F) or (M,) array of per-tower costs
        R_0 : ndarray
            (M,F) or (M,1) array of initial guess for radii
        p_0 : ndarray
            (J*F,) or (J,) array of initial guess for prices
        symmetric : bool
            specifies whether the equilibrium solving for is symmetric (quicker to compute)
        print_msg : bool
            determines whether or not to print inputs and output of root solver
        impute_MVNO : dict
            dict with
            'impute' : bool (whether to impute the Qs for MVNO)
            'firms_share' (optional) : ndarray ((F-1,) array of whether firms share qualities with MVNOs)
            'include' (optional) : bool (whether to include MVNO Q in returned Q)
        q_0 : ndarray
            (M,F) array of initial guess of q
        eps_R : float
            size of perturbation to measure radius derivative
        eps_p : float
            size of perturbation to measure price derivative
        eps_bw : float
            size of perturbation to measure bandwidth derivative
        factor : float
            size of the factor for fsolve, must be in interval [0.1, 100]
        include_logit_shock : bool
            determine whether or not to include logit shocks in the consumer surplus calculation
        adjust_c_R : bool
            determine whether or not to adjust c_R with bandwidth

    Returns
    -------
        partial_Pif_partial_bf : ndarray
            (M,F) array of derivative 
        partial_Pif_partial_b : ndarray
            (M,F) array of firms' optimal prices
        partial_CS_partial_b : float
            qualities that result from prices and infrastructure
        successful : bool
            True if all equilibrium computations successful, False otherwise
    """
    
    # Add indices
    pidx = ds.chars.index(ds.pname)
    qidx = ds.chars.index(ds.qname)
    
    # Create high and low bandwidth arrays
    bw_high = bw + eps_bw
    bw_low = bw - eps_bw
    
    successful = True
    
    if adjust_c_R:
        c_R_high = c_R / bw * bw_high
        c_R_low = c_R / bw * bw_low
    else:
        c_R_high = np.copy(c_R)
        c_R_low = np.copy(c_R)
    
    # Determine derivative of increasing a firm's bandwidth on its profits
    partial_Pif_partial_bf = np.zeros(R_0.shape)
    firms, firm_counts = np.unique(ds.firms, return_counts=True)
    expand_firms = lambda x: np.tile(x, (1,firms.shape[0] if symmetric else 1))
    expand_firms_1d = lambda x: np.tile(x, (firms.shape[0] if symmetric else 1,))
    for f in range(partial_Pif_partial_bf.shape[1]):
        # Create bandwidth arrays
        bw_high_f = expand_firms(bw)
        bw_high_f[:,f] = bw_high[:,f]
        bw_low_f = expand_firms(bw)
        bw_low_f[:,f] = bw_low[:,f]
        c_R_high_f = expand_firms(c_R)
        c_R_low_f = expand_firms(c_R)
        c_R_high_f[:,f] = c_R_high[:,f]
        c_R_low_f[:,f] = c_R_low[:,f]
        
        # Determine equilibrium for high and low bandwidths
        R_stars_high, p_stars_high, q_stars_high, success_high = infrastructure_eqm(bw_high_f, gamma, ds, expand_firms(xis), theta, pop, market_size, expand_firms_1d(c_u), c_R_high_f, expand_firms(R_0), expand_firms_1d(p_0), symmetric=False, print_msg=print_msg, impute_MVNO=impute_MVNO, q_0=q_0, eps_R=eps_R, eps_p=eps_p, factor=factor)
        R_stars_low, p_stars_low, q_stars_low, success_low = infrastructure_eqm(bw_low_f, gamma, ds, expand_firms(xis), theta, pop, market_size, expand_firms_1d(c_u), c_R_low_f, expand_firms(R_0), expand_firms_1d(p_0), symmetric=False, print_msg=print_msg, impute_MVNO=impute_MVNO, q_0=q_0, eps_R=eps_R, eps_p=eps_p, factor=factor)
        
        # Add equilibrium results to DemandSystem
        ds_high = copy.deepcopy(ds)
        ds_high.data[:,:,pidx] = p_stars_high[np.newaxis,:]
        ds_high.data[:,:,qidx] = np.repeat(q_stars_high, firm_counts, axis=1) # only works b/c products in order
        ds_low = copy.deepcopy(ds)
        ds_low.data[:,:,pidx] = p_stars_low[np.newaxis,:]
        ds_low.data[:,:,qidx] = np.repeat(q_stars_low, firm_counts, axis=1) # only works b/c products in order
        
        # Determine impact on per-user profit
        shares_high = blp.s_mj(ds_high, theta, ds_high.data, expand_firms(xis)) * pop[:,np.newaxis]
        profits_high = np.sum((shares_high * (p_stars_high - expand_firms_1d(c_u))[np.newaxis,:])[:,ds.firms == firms[f]], axis=1)
        stations_cost_high = (infr.num_stations(R_stars_high, market_size[:,np.newaxis]) * c_R_high_f)[:,f]
        Pif_high = (profits_high - stations_cost_high) / pop
        shares_low = blp.s_mj(ds_low, theta, ds_low.data, expand_firms(xis)) * pop[:,np.newaxis]
        profits_low = np.sum((shares_low * (p_stars_low - expand_firms_1d(c_u))[np.newaxis,:])[:,ds.firms == firms[f]], axis=1)
        stations_cost_low = (infr.num_stations(R_stars_low, market_size[:,np.newaxis]) * c_R_low_f)[:,f]
        Pif_low = (profits_low - stations_cost_low) / pop
        
        # Determine partial derivative
        partial_Pif_partial_bf[:,f] = (Pif_high - Pif_low) / (2. * eps_bw)
        if not success_high or not success_low:
            successful = False
        
    # Determine derivative of increasing all firms bandwidth on an individual firm's profits
    partial_Pif_partial_b = np.zeros(R_0.shape)
    # Determine equilibrium for high and low bandwidths
    R_stars_high, p_stars_high, q_stars_high, success_high = infrastructure_eqm(bw_high, gamma, ds, xis, theta, pop, market_size, c_u, c_R_high, R_0, p_0, symmetric=symmetric, print_msg=print_msg, impute_MVNO=impute_MVNO, q_0=q_0, eps_R=eps_R, eps_p=eps_p, factor=factor)
    R_stars_low, p_stars_low, q_stars_low, success_low = infrastructure_eqm(bw_low, gamma, ds, xis, theta, pop, market_size, c_u, c_R_low, R_0, p_0, symmetric=symmetric, print_msg=print_msg, impute_MVNO=impute_MVNO, q_0=q_0, eps_R=eps_R, eps_p=eps_p, factor=factor)

    # Add equilibrium results to DemandSystem
    ds_high = copy.deepcopy(ds)
    ds_high.data[:,:,pidx] = p_stars_high[np.newaxis,:]
    ds_high.data[:,:,qidx] = np.repeat(q_stars_high, firm_counts, axis=1) # only works b/c products in order
    ds_low = copy.deepcopy(ds)
    ds_low.data[:,:,pidx] = p_stars_low[np.newaxis,:]
    ds_low.data[:,:,qidx] = np.repeat(q_stars_low, firm_counts, axis=1) # only works b/c products in order

    # Determine impact on per-user profit
    for f in range(partial_Pif_partial_b.shape[1]):
        shares_high = blp.s_mj(ds_high, theta, ds_high.data, expand_firms(xis)) * pop[:,np.newaxis]
        profits_high = np.sum((shares_high * (p_stars_high - expand_firms_1d(c_u))[np.newaxis,:])[:,ds.firms == firms[f]], axis=1)
        stations_cost_high = (infr.num_stations(R_stars_high, market_size[:,np.newaxis]) * c_R_high)[:,f]
        Pif_high = (profits_high - stations_cost_high) / pop
        shares_low = blp.s_mj(ds_low, theta, ds_low.data, expand_firms(xis)) * pop[:,np.newaxis]
        profits_low = np.sum((shares_low * (p_stars_low - expand_firms_1d(c_u))[np.newaxis,:])[:,ds.firms == firms[f]], axis=1)
        stations_cost_low = (infr.num_stations(R_stars_low, market_size[:,np.newaxis]) * c_R_low)[:,f]
        Pif_low = (profits_low - stations_cost_low) / pop
        
        # Determine partial derivative
        partial_Pif_partial_b[:,f] = (Pif_high - Pif_low) / (2. * eps_bw)

    # Determine impact on consumer surplus
    CS_high = np.mean(welfare.consumer_surplus(ds_high, expand_firms(xis), theta, include_logit_shock=include_logit_shock), axis=1)
    CS_low = np.mean(welfare.consumer_surplus(ds_low, expand_firms(xis), theta, include_logit_shock=include_logit_shock), axis=1)

    # Determine partial derivative
    partial_CS_partial_b = (CS_high - CS_low) / (2. * eps_bw)
    if not success_high or not success_low:
        successful = False
    
    return partial_Pif_partial_bf, partial_Pif_partial_b, partial_CS_partial_b, successful
