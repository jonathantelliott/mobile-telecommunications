# %%
import numpy as np
#import numpy as np
from scipy import integrate

# %%
# Global parameters
freq_rep = 1900. # representative frequency
height_rep = 30. # representative height of base station
A0 = 61. # baseline signal strength in dBm / 5 MHz
JN_noise = -174. + 10. * np.log10(5.e6) # Johnson-Nyquist noise per 5 MHz
cell_utilization = 0.02

# %%
def hata_loss(radius, freq, height):
    """
        Return small city Hata loss function
    
    Parameters
    ----------
        radius : float
            cell radius in km
        freq : float 
            frequency in MHz
        height : float
            antenna height in m

    Returns
    -------
        path_loss: float
            scalar of path loss
    """

    path_loss =  69.55 + 26.16 * np.log10(freq) - 13.82 * np.log10(height) - 0.8 + 1.56 * np.log10(freq) + (44.9 - 6.55 * np.log10(height)) * np.log10(radius)

    return path_loss

def loss_power_ratio(radius, freq, height):
    """
        Return loss in terms of power ratio
    
    Parameters
    ----------
        radius : float
            cell radius in km
        freq : float 
            frequency in MHz
        height : float
            antenna height in m

    Returns
    -------
        loss_pr : float 
            path loss power ratio
    """

    loss_pr = np.exp(hata_loss(radius, freq, height) * np.log(10.) / 10.)

    return loss_pr

def interfere_power(A, radius, freq, height):
    """
        Return interference from adjacent six cells
    
    Parameters
    ----------
        A : float
            power in dBm / 5 MHz
        radius : float
            cell radius in km
        freq : float 
            frequency in MHz
        height : float
            antenna height in m

    Returns
    -------
        interference : float
            interference from adjacent six cells, in dBm
    """

    loss = hata_loss(radius * np.sqrt(3), freq, height)
    interference = A - loss + 10. * np.log10(6. * cell_utilization)

    return interference

def SINR(A, radius, freq, height):
    """
        Return signal-to-interference-and-noise ratio
    
    Parameters
    ----------
        A : float
            power in dBm / 5 MHz
        radius : float
            cell radius in km
        freq : float 
            frequency, in MHz
        height : float
            antenna height, in m

    Returns
    -------
        ratio : float
            signal-to-interference-and-noise ratio, power ratio
    """

    loss = hata_loss(radius, freq, height)
    interference = interfere_power(A, radius, freq, height)
    signal_power = 10.**((A - loss) / 10.)
    noise_power = 10.**(JN_noise / 10.)
    interference_power = 10.**(interference / 10.)
    ratio = signal_power / (noise_power + interference_power)

    return ratio

def rho_C_hex(bw, radius, gamma):
    """
        Return total transmission capacity of station, assuming transmission evenly distributed over hexagonal cell
    
    Parameters
    ----------
        bw : float
            bandwidth in MHz
        radius : float
            cell radius in km
        gamma : float 
            spectral efficiency

    Returns
    -------
        channel_cap : float
            transmission capacity of station, in Mbps, based on Shannon-Hartley Theorem
    """

    if bw <= 0:
        channel_cap = 0
    else:
        cell_area = 3. * np.sqrt(3.) / 2. * (radius**2. - 0.001**2.)
        num_triangles = 6. * 2.
        transmission = lambda s, r: num_triangles / np.log2(1. + SINR(A0, np.sqrt(s**2. + r**2.), freq_rep, height_rep))
        mean_transmission = integrate.dblquad(transmission, 0.001 * np.sqrt(3./4.), radius * np.sqrt(3./4.), lambda r: 0, lambda r: r / np.sqrt(3.))[0] # take only the first argument (result), not second (error estimate)
        channel_cap = gamma * cell_area * bw / mean_transmission

    return channel_cap

def num_stations(R, market_size):
    """
        Return the number of stations associated with radius
    
    Parameters
    ----------
        R : ndarray
            array of radii
        market_size : ndarray
            array of geographic size of markets in km^2

    Returns
    -------
        stations : ndarray
            array of number of stations associated with radii in each market
    """

    cell_area = 3. * np.sqrt(3.) / 2. * R**2. # based on hexagonal cells
    stations = market_size / cell_area

    return stations

def num_stations_deriv(R, market_size):
    """
        Return the derivative of the number of stations associated with radius
    
    Parameters
    ----------
        R : ndarray
            array of radii
        market_size : ndarray
            array of geographic size of markets in km^2

    Returns
    -------
        deriv : ndarray
            array of derivative of number of stations associated with radii in each market
    """

    deriv = -2. * market_size / (3. * np.sqrt(3.) / 2. * R**3.)

    return deriv

# %%
# Test
# import matplotlib.pyplot as plt

# Rs = np.linspace(0., 15., 100)
# rhos = np.zeros(100)
# for i, R in enumerate(Rs):
#     rhos[i] = rho_C_hex(0.1, R, 1.)

# plt.plot(Rs, rhos)
# plt.show()

# %%
