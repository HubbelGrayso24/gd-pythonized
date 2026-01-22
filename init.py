import scipy.io as sio
import numpy as np

def initialize(load_trmult):
    global H0, a, m2, C_vect, tau0, pop0, pop5, pop5_fertadj, popminus5, popminus10, ubar
    global trmult_reduced, n, earth_indices, indicator_sea, subs, beta, tail_bands, ind_islands
    global alpha, theta, Omega, vect_omega

    # 1. Load initial land, amenities, productivity, moving costs, and population
    H0_data = sio.loadmat('H0.mat')
    global H0
    H0 = H0_data['H0']

    a_H0_data = sio.loadmat('a_H0.mat')
    global a
    a = a_H0_data['a_H0']
    
    # Normalize `a` (Assuming `a_norm` is the normalized version of `a`)
    global a_norm
    a_norm = (a - np.min(a)) / (np.max(a) - np.min(a))  # Normalization example

    tau_H0_data = sio.loadmat('tau_H0.mat')
    global tau0
    tau0 = tau_H0_data['tau_H0']

    m2_data = sio.loadmat('m2.mat')
    global m2
    m2 = m2_data['m2']

    global pop0, pop5, pop5_fertadj, popminus5, popminus10
    pop0 = np.loadtxt('l.csv', delimiter=',')
    pop5 = np.loadtxt('pop5.csv', delimiter=',')
    popminus5 = np.loadtxt('popminus5.csv', delimiter=',')
    popminus10 = np.loadtxt('popminus10.csv', delimiter=',')
    pop5_fertadj = np.loadtxt('pop5_fertadj.csv', delimiter=',')

    # 2. Find coordinates of emerged cells on the coarse grid
    global earth_indices, n, indicator_sea
    earth_indices = np.where(H0 > 0)[0]
    n = len(earth_indices)
    indicator_sea = (H0 == 0)

    # 3. Load and shape utility levels
    global ubar
    ubar = np.loadtxt('ubar.csv', delimiter=',')

    # Remove NaNs and Infs from ubar
    ubar[np.isnan(ubar)] = 0
    ubar[np.isinf(ubar)] = 0

    # 4. Load trmult or not
    global trmult_reduced
    if load_trmult == 1:
        trmult_reduced_data = sio.loadmat('trmult_reduced.mat')
        trmult_reduced = trmult_reduced_data['trmult_reduced']
    else:
        trmult_reduced = None  # Set to None if not loaded

    # 5. Load and Manipulate countries to get rid of missing indices
    C = np.loadtxt('C.csv', delimiter=',')
    
    C_stock = C[earth_indices]
    indices = np.unique(C_stock)
    C_stock_2 = C_stock.copy()
    
    for i in range(len(indices)):
        C_stock_2[C_stock == indices[i]] = i + 1
    
    C[earth_indices] = C_stock_2

    # subs contains the indicators of the countries (+1 to avoid zero).
    # it can be used later to create maps aggregated per country
    # C_vect contains the index of each emerged cell's country
    global subs, C_vect
    subs = C.reshape(-1) + 1
    C_vect = C[earth_indices]

    # 6. Set other global variables
    global beta, tail_bands, alpha, theta, Omega
    beta = 0.965
    tail_bands = 0.2
    alpha = 0.06
    theta = 6.5
    Omega = 0.5

    # Return all variables in the specified order
    results = [
        H0, a, a_norm, m2, C_vect, tau0, pop0, pop5, pop5_fertadj, popminus5, popminus10, ubar,
        trmult_reduced, n, earth_indices, indicator_sea, subs, None, beta, tail_bands, None, alpha, theta, Omega
    ]

    return results
