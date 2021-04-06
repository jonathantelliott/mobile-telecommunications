# %%
import numpy as np
from scipy.optimize import fsolve

import counterfactuals.infrastructurefunctions as infr

import demand.dataexpressions as de
import demand.blpextension as blp

# %%
def avg_Q(q_S, q_D):
    """
        Return total transmission capacity of station, assuming transmission evenly distributed over hexagonal cell
    
    Parameters
    ----------
        q_S : float
            channel capacity in Mbps
        q_D : float
            data demand rate in Mbps
            

    Returns
    -------
        q : float
            average demand speed, in Mbps, based on Shannon-Hartley Theorem
    """

    q = q_S - q_D

    return q

def data_demand(ds, xis, theta, Q, pop):
    """
        Return the data demanded in a month
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about our markets
        xis : ndarray 
            (M,J) matrix of vertical demand components
        theta : ndarray
            (K,) array of demand parameters
        Q : ndarray
            (M,F) array of market-firm-specific download speeds (in Mbps)
        pop : ndarray
            (M,) array of market populations

    Returns
    -------
        predicted_dbar : ndarray
            (M,F) array of data demanded in month (in MB)
    """

    # Process data
    X = np.copy(ds.data)
    qidx = ds.chars.index(ds.qname)
    firms, firm_counts = np.unique(ds.firms, return_counts=True)
    Q_expand = np.repeat(Q, firm_counts, axis=1)
    X[:,:,qidx] = Q_expand
    dlimidx = ds.chars.index(ds.dlimname)
    dlim = X[:,:,dlimidx]

    # Calculate data consumption of each type
    Ex = de.E_x(ds, theta, X, Q_expand, dlim, blp.ycX(ds, theta, X)) # M x J x I, this is in GB
    Ex = Ex * de.conv_factor # convert from GB to MB

    # Aggregate data consumption, weighting by shares
    s_ijm = blp.s_mji(ds, theta, X, xis) # M x J x I
    # calculate weights from the shares of adoption of product j by i times weight of i
    num_i = s_ijm.shape[2]
    weights = s_ijm * (np.ones(num_i) / num_i)[np.newaxis,np.newaxis,:] # only works b/c quantiles, uniformly distributed
    predicted_dbar_avg = np.sum(Ex * weights, axis=2) / np.sum(weights, axis=2) # weighted average across i
    predicted_dbar = predicted_dbar_avg * pop[:,np.newaxis]

    return predicted_dbar

def data_demand_rate(ds, xis, theta, Q, num_stations, pop):
    """
        Return the data demand rate
    
    Parameters
    ----------
        ds : DemandSystem
            contains all the data about our markets
        xis : ndarray 
            (M,J) matrix of vertical demand components
        theta : ndarray
            (K,) array of demand parameters
        Q : ndarray
            (M,F) array of market-firm-specific download speeds (in Mbps)
        num_stations : ndarray
            (M,F) array of number of stations in each market
        pop : ndarray
            (M,) array of market populations

    Returns
    -------
        Q_D : ndarray
            (M,F) array of data demand rate (in Mbps)
    """

    predicted_dbar_j = data_demand(ds, xis, theta, Q, pop)
    
    # Aggregate to firm-level
    firms = np.unique(ds.firms)
    predicted_dbar_f = np.zeros((predicted_dbar_j.shape[0], firms.shape[0]))
    for i, firm in enumerate(firms):
        predicted_dbar_f[:,i] = np.sum(predicted_dbar_j[:,ds.firms == firm], axis=1)
    
    # Turn data demanded over month to demand rate
    num_hours_in_day = 24. #6. # people only use phone during the day
    num_seconds_month = 60. * 60. * num_hours_in_day * 30.
    byte_to_bit_conv = 8.
    Q_D = byte_to_bit_conv * predicted_dbar_f / num_stations / num_seconds_month # Mb per station per second

    return Q_D

def q_MVNO(qs, firms_share):
    """
        Return the MVNO quality as a function of other firms qualities
    
    Parameters
    ----------
        qs : ndarray
            (M,F-1) array of market-firm-specific download speeds in Gbps
        firms_share : ndarray
            (F-1,) array of whether firms share qualities with MVNOs

    Returns
    -------
        qs_MVNO : ndarray
            (M,) array of imputed MVNO qualities
    """

    return np.mean(qs[:,firms_share], axis=1)

def q_res(q, cc, ds, xis, theta, num_stations, pop, impute_MVNO={'impute': False}):
    """
        Return the residual relative to predicted quality
    
    Parameters
    ----------
        q : ndarray
            (M,F-1) array of market-firm-specific download speeds in Mbps
        cc : ndarray
            (M,F-1) array of channel capacity in Mbps
        ds : DemandSystem
            contains all the data about our markets
        xis : ndarray 
            (M,J) matrix of vertical demand components
        theta : ndarray
            (K,) array of demand parameters
        num_stations : ndarray
            (M,F-1) array of number of stations in each market
        pop : ndarray
            (M,) array of market populations
        impute_MVNO : dict
            dict with
            'impute' : bool (whether)
            'firms_share' (optional) : ndarray ((F-1,) array of whether firms share qualities with MVNOs)

    Returns
    -------
        res : ndarray
            (M,F-1) array of residual relative to predicted quality (in Mbps)
    """

    # Determine the data demand rate
    if impute_MVNO['impute']:
        qs_use = np.concatenate((q, q_MVNO(q, impute_MVNO['firms_share'])[:,np.newaxis]), axis=1) # impute MVNO quality
    else:
        qs_use = q
    Q_D = data_demand_rate(ds, xis, theta, qs_use, num_stations, pop)

    # Solve for what the data demand rate implies the quality must be for the four MNOs
    Q = np.zeros(q.shape)
    for m in range(q.shape[0]):
        for f in range(q.shape[1]): # for all firms, except MVNOs
            Q[m,f] = avg_Q(cc[m,f], Q_D[m,f])

    res = q - Q
    
    return res

def q(cc, ds, xis, theta, num_stations, pop, impute_MVNO={'impute': False}, q_0=None):
    """
        Return the equilibrium quality
    
    Parameters
    ----------
        cc : ndarray
            (M,F-1) array of channel capacities in Mb/s
        ds : DemandSystem
            contains all the data about our markets
        xis : ndarray 
            (M,J) matrix of vertical demand components
        theta : ndarray
            (K,) array of demand parameters
        num_stations : ndarray
            (M,F-1) array of number of stations in each market
        pop : ndarray
            (M,) array of market populations
        impute_MVNO : dict
            dict with
            'impute' : bool (whether to impute the Qs for MVNO)
            'firms_share' (optional) : ndarray ((F-1,) array of whether firms share qualities with MVNOs)
            'include' (optional) : bool (whether to include MVNO Q in returned Q)
        q_0 : ndarray
            (M,F-1) array of initial guess of q

    Returns
    -------
        q_star : ndarray
            (M,F) array of market-firm-specific download speeds in Mbps
    """
    
    # Create a starting guess for q_0 if none provided
    if q_0 is None:
        firms, firms_idx = np.unique(ds.firms, return_index=True)
        if impute_MVNO['impute']:
            firms_idx = firms_idx[:-1] # don't want to include MVNO if imputing, uses fact that MVNO is last firm
        qidx = ds.chars.index(ds.qname)
        # q_0 = ds.data[:,firms_idx,qidx]
        q_0 = np.ones(cc.shape) * 0.001 # use this one instead; otherwise, it will move into negative 
    
    # Add on num_stations for MVNOs if MVNOs included and imputed
    if impute_MVNO['impute']:
        num_stations_use = np.concatenate((num_stations, np.ones((num_stations.shape[0],1)) * np.nan), axis=1) # add a vector of NaNs to MVNO num_stations, this column dropped later so doesn't matter
    else:
        num_stations_use = num_stations
    
    # Solve for qs that satisfy transmission equilibrium
    q_eq = lambda qs: np.reshape(q_res(np.reshape(qs, cc.shape), cc, ds, xis, theta, num_stations_use, pop, impute_MVNO), (-1,))
    q_star, infodict, ier, msg = fsolve(q_eq, np.reshape(q_0, (-1,)), full_output=True) # could add Jacobian, a bit more difficult
    q_star = np.reshape(q_star, cc.shape)
    
    # Print error message if failed to converge
    if ier != 1:
        print(f"Transmission equilibrium computation failed for following reason: {msg}. Additional information: {infodict}")
    
    # Add MVNOs if imputing MVNO
    if impute_MVNO['impute']:
        if impute_MVNO['include']:
            q_star = np.concatenate((q_star, q_MVNO(q_star, impute_MVNO['firms_share'])[:,np.newaxis]), axis=1)

    # Return qualities
    return q_star
