import numpy as np
import scipy.io as sio

def _load_mat_any(path):
    try:
        return sio.loadmat(path)
    except NotImplementedError:
        import h5py

        out = {}
        with h5py.File(path, 'r') as f:
            for k in f.keys():
                out[k] = np.array(f[k])
        return out


def _load_mat_var(path, var_name=None):
    d = _load_mat_any(path)
    if var_name is not None:
        if var_name not in d:
            raise KeyError(f"Variable '{var_name}' not found in {path}. Keys: {list(d.keys())}")
        return d[var_name]

    keys = [k for k in d.keys() if not k.startswith('__')]
    if len(keys) != 1:
        raise KeyError(f"Expected exactly one variable in {path}; found keys: {keys}")
    return d[keys[0]]


def _load_txt_flat(path):
    arr = np.loadtxt(path, delimiter=',')
    return np.asarray(arr).reshape(-1)


def initialize(load_trmult):
    global H0, a, a_norm, m2, C_vect, tau0, pop0, pop5, pop5_fertadj, popminus5, popminus10, ubar
    global trmult_reduced, n, earth_indices, indicator_sea, subs, beta, tail_bands, ind_islands
    global alpha, theta, Omega, vect_omega

    try:
        H0 = _load_mat_var('Data/H0.mat', 'H0')
    except FileNotFoundError:
        H0 = _load_mat_var('H0.mat', 'H0')

    a = _load_mat_var('Data/a_H0.mat', 'a_H0')
    tau0 = _load_mat_var('Data/tau_H0.mat', 'tau_H0')
    m2 = _load_mat_var('Data/m2.mat', 'm2')

    a = np.asarray(a).reshape(-1)
    tau0 = np.asarray(tau0).reshape(-1)
    m2 = np.asarray(m2).reshape(-1)

    a_norm = None

    pop0 = _load_txt_flat('Data/l.csv')
    pop5 = _load_txt_flat('Data/pop5.csv')
    popminus5 = _load_txt_flat('Data/popminus5.csv')
    popminus10 = _load_txt_flat('Data/popminus10.csv')
    pop5_fertadj = _load_txt_flat('Data/pop5_fertadj.csv')

    H0_arr = np.asarray(H0)
    earth_indices = np.flatnonzero(H0_arr.reshape(-1) > 0)
    n = int(earth_indices.size)
    indicator_sea = (H0_arr == 0)

    ubar = _load_txt_flat('Data/ubar.csv')
    ubar[np.isnan(ubar)] = 0
    ubar[np.isinf(ubar)] = 0

    if load_trmult == 1:
        trmult_reduced = _load_mat_var('Data/trmult_reduced.mat', 'trmult_reduced')
        trmult_reduced = np.asarray(trmult_reduced)
        trmult_reduced[trmult_reduced < 1e-12] = 0
    else:
        trmult_reduced = None

    C = _load_txt_flat('Data/C.csv')
    C_stock = C[earth_indices]
    indices = np.unique(C_stock)
    C_stock_2 = C_stock.copy()
    for i, idx in enumerate(indices, start=1):
        C_stock_2[C_stock == idx] = i
    C[earth_indices] = C_stock_2

    subs = C.reshape(-1) + 1
    C_vect = C[earth_indices]

    beta = 0.965
    tail_bands = 0.2
    alpha = 0.06
    theta = 6.5
    Omega = 0.5

    results = [
        H0, a, a_norm, m2, C_vect, tau0, pop0, pop5, pop5_fertadj, popminus5, popminus10, ubar,
        trmult_reduced, n, earth_indices, indicator_sea, subs, None, beta, tail_bands, None, alpha, theta, Omega
    ]
    return results
