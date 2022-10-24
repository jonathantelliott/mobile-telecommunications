# %%
import numpy as np
#import numpy as np
from scipy import integrate

# %%
# Global parameters
freq_rep = 1900. # representative frequency
height_rep = 30. # representative height of base station
A0 = 61. # baseline signal strength in dBm / 5 MHz
JN_noise = -174. + 10. * np.log10(5.0e6) # Johnson-Nyquist noise per 5 MHz
cell_utilization = 0.3 # from Blaszczyszyn et al., 2014

# %%
def hata_loss(radius, freq, height):
    """
        Return small city Hata loss function
    
    Parameters
    ----------
        radius : float or ndarray
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

def interfere_power(A, x, y, radius, freq, height):
    """
        Return interference from adjacent cells
    
    Parameters
    ----------
        A : float
            power in dBm / 5 MHz
        x : float
            coordinate x in km
        y : float
            coordinate y in km
        radius : float
            size of radius in km
        freq : float 
            frequency in MHz
        height : float
            antenna height in m

    Returns
    -------
        interference : ndarray
            interference from six adjacent cells, in dBm
    """
    
    # Construct x and y coordinates from the perspective of the six adjacent cells
    dist_x = np.array([0., 3. / 2., 3. / 2., 0., 3. / 2., 3. / 2.]) * radius
    dist_y = np.array([np.sqrt(3.), np.sqrt(3.) / 2., np.sqrt(3.) / 2., np.sqrt(3.), np.sqrt(3.) / 2., np.sqrt(3.) / 2.]) * radius
    direction_x = np.array([1., -1., -1., 1., 1., 1.])
    direction_y = np.array([-1., -1., 1., 1., 1., -1.])
    x_coords = dist_x + direction_x * x
    y_coords = dist_y + direction_y * y
    
    # Determine interference power from the six adjacent cells
    radii = np.sqrt(x_coords**2. + y_coords**2.)
    interference = A - hata_loss(radii, freq, height)

    return interference

def SINR(A, x, y, radius, freq, height):
    """
        Return signal-to-interference-and-noise ratio
    
    Parameters
    ----------
        A : float
            power in dBm / 5 MHz
        x : float
            coordinate x in km
        y : float
            coordinate y in km
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

    signal_power = 10.**((A - hata_loss(np.sqrt(x**2. + y**2.), freq, height)) / 10.)
    noise_power = 10.**(JN_noise / 10.)
    interference_power = np.sum(cell_utilization * 10.**(interfere_power(A, x, y, radius, freq, height) / 10.)) # cell only utilized cell_utilization fraction of the time
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
        cell_area = 3. * np.sqrt(3.) / 2. * radius**2.
        num_triangles = 6. * 2.
        transmission = lambda x, y: num_triangles / np.log2(1. + SINR(A0, x, y, radius, freq_rep, height_rep))
        mean_transmission = integrate.dblquad(transmission, 0., radius * np.sqrt(3./4.), lambda y: 0., lambda y: y / np.sqrt(3.))[0] # take only the first argument (result), not second (error estimate)
        channel_cap = gamma * cell_area * bw / mean_transmission

    return channel_cap

def avg_path_loss(radius):
    """
        Return average path loss over the cell
    
    Parameters
    ----------
        radius : float
            cell radius in km

    Returns
    -------
        mean_path_loss : float
            average path loss experienced by consumers in cell
    """

    cell_area = 3. * np.sqrt(3.) / 2. * radius**2.
    num_triangles = 6. * 2.
    path_loss = lambda y, x: num_triangles * hata_loss(np.sqrt(x**2. + y**2.), freq_rep, height_rep)
    mean_path_loss = 1.0 / cell_area * integrate.dblquad(path_loss, 0., radius * np.sqrt(3./4.), lambda y: 0., lambda y: y / np.sqrt(3.))[0] # take only the first argument (result), not second (error estimate)

    return mean_path_loss

def avg_SINR(radius):
    """
        Return average SINR over the cell
    
    Parameters
    ----------
        radius : float
            cell radius in km

    Returns
    -------
        mean_SINR : float
            average SINR experienced by consumers in cell
    """

    cell_area = 3. * np.sqrt(3.) / 2. * radius**2.
    num_triangles = 6. * 2.
    SINR_ = lambda y, x: num_triangles * SINR(A0, x, y, radius, freq_rep, height_rep)
    mean_SINR = 1.0 / cell_area * integrate.dblquad(SINR_, 0., radius * np.sqrt(3./4.), lambda y: 0., lambda y: y / np.sqrt(3.))[0] # take only the first argument (result), not second (error estimate)

    return mean_SINR

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
