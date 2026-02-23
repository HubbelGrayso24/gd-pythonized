import numpy as np
import scipy.io as sio
from pathlib import Path

from init import initialize
from results import results
from backward import backward
from plots import plots


global H0, a, a_norm, m2, C_vect, tau0, pop0, pop5, pop5_fertadj, popminus5, popminus10, ubar, trmult_reduced, n, earth_indices, indicator_sea, subs, subs_vect, beta, tail_bands, ind_islands, alpha, theta, Omega

def calculate_correlation(x, y):
    return np.corrcoef(x, y)[0, 1]


def run_model(nb_per=600, nb_back=180, save_output=True, make_plots=True, curledtheta=1.0):
    # Initialize model
    vars = initialize(1)
    H0, a, a_norm, m2, C_vect, tau0, pop0, pop5, pop5_fertadj, popminus5, popminus10, ubar, trmult_reduced, n, earth_indices, indicator_sea, subs, subs_vect, beta, tail_bands, ind_islands, alpha, theta, Omega = vars

    vars_for_run = vars
    if curledtheta is not None:
        vars_list = list(vars)
        m2_cf = np.asarray(m2).reshape(-1) ** float(curledtheta)
        vars_list[3] = m2_cf
        vars_for_run = tuple(vars_list)

    # Distribution of land for simulation
    H0_arr = np.asarray(H0).reshape(-1)
    H = H0_arr[earth_indices]

    # Run the model and obtain summary statistics
    results_data = results(H, nb_per, vars_for_run)
    realgdp_w, u_w, u2_w, prod_w, phi_w, PDV_u_w, PDV_u2_w, PDV_realgdp_w, migr_cell, migr_ctry, l, u, u2, tau, realgdp = results_data

    # Plot time series and maps, and save them
    if make_plots:
        plots(H, realgdp_w, u_w, u2_w, prod_w, l, u, tau, realgdp)

    # Run model backwards
    l_b, u_b, w_b, tau_b, phi_b, realgdp_b = backward(H, nb_back, vars_for_run)

    print('CORRELATIONS WITH 1995 DATA - CELL LEVEL')
    print(calculate_correlation(popminus5[earth_indices], H0_arr[earth_indices] * l_b[:, 4]))
    print(calculate_correlation(np.log(popminus5[earth_indices]), np.log(H0_arr[earth_indices] * l_b[:, 4])))
    print(calculate_correlation(pop0[earth_indices] - popminus5[earth_indices], pop0[earth_indices] - H0_arr[earth_indices] * l_b[:, 4]))
    print(calculate_correlation(np.log(pop0[earth_indices]) - np.log(popminus5[earth_indices]), np.log(pop0[earth_indices]) - np.log(H0_arr[earth_indices] * l_b[:, 4])))

    print('CORRELATIONS WITH 1990 DATA - CELL LEVEL')
    print(calculate_correlation(popminus10[earth_indices], H0_arr[earth_indices] * l_b[:, 9]))
    print(calculate_correlation(np.log(popminus10[earth_indices]), np.log(H0_arr[earth_indices] * l_b[:, 9])))
    print(calculate_correlation(pop0[earth_indices] - popminus10[earth_indices], pop0[earth_indices] - H0_arr[earth_indices] * l_b[:, 9]))
    print(calculate_correlation(np.log(pop0[earth_indices]) - np.log(popminus10[earth_indices]), np.log(pop0[earth_indices]) - np.log(H0_arr[earth_indices] * l_b[:, 9])))

    print('CORRELATIONS WITH 1995 DATA - COUNTRY LEVEL')
    ctry_idx = C_vect.astype(int) - 1
    popminus5_ctry_d = np.bincount(ctry_idx, weights=popminus5[earth_indices])
    popminus5_ctry_m = np.bincount(ctry_idx, weights=H0_arr[earth_indices] * l_b[:, 4])
    pop0_ctry = np.bincount(ctry_idx, weights=pop0[earth_indices])
    print(calculate_correlation(popminus5_ctry_d, popminus5_ctry_m))
    print(calculate_correlation(np.log(popminus5_ctry_d), np.log(popminus5_ctry_m)))
    print(calculate_correlation(pop0_ctry - popminus5_ctry_d, pop0_ctry - popminus5_ctry_m))
    print(calculate_correlation(np.log(pop0_ctry) - np.log(popminus5_ctry_d), np.log(pop0_ctry) - np.log(popminus5_ctry_m)))

    print('CORRELATIONS WITH 1990 DATA - COUNTRY LEVEL')
    popminus10_ctry_d = np.bincount(ctry_idx, weights=popminus10[earth_indices])
    popminus10_ctry_m = np.bincount(ctry_idx, weights=H0_arr[earth_indices] * l_b[:, 9])
    print(calculate_correlation(popminus10_ctry_d, popminus10_ctry_m))
    print(calculate_correlation(np.log(popminus10_ctry_d), np.log(popminus10_ctry_m)))
    print(calculate_correlation(pop0_ctry - popminus10_ctry_d, pop0_ctry - popminus10_ctry_m))
    print(calculate_correlation(np.log(pop0_ctry) - np.log(popminus10_ctry_d), np.log(pop0_ctry) - np.log(popminus10_ctry_m)))

    if save_output:
        Path('Output').mkdir(parents=True, exist_ok=True)
        sio.savemat('Output/realgdp_w.mat', {'realgdp_w': realgdp_w})
        sio.savemat('Output/u_w.mat', {'u_w': u_w})
        sio.savemat('Output/u2_w.mat', {'u2_w': u2_w})
        sio.savemat('Output/prod_w.mat', {'prod_w': prod_w})
        sio.savemat('Output/phi_w.mat', {'phi_w': phi_w})
        sio.savemat('Output/PDV_u_w.mat', {'PDV_u_w': PDV_u_w})
        sio.savemat('Output/PDV_u2_w.mat', {'PDV_u2_w': PDV_u2_w})
        sio.savemat('Output/PDV_realgdp_w.mat', {'PDV_realgdp_w': PDV_realgdp_w})
        sio.savemat('Output/migr_cell.mat', {'migr_cell': migr_cell})
        sio.savemat('Output/migr_ctry.mat', {'migr_ctry': migr_ctry})
        sio.savemat('Output/l.mat', {'l': l})
        sio.savemat('Output/u.mat', {'u': u})
        sio.savemat('Output/realgdp.mat', {'realgdp': realgdp})
        sio.savemat('Output/tau.mat', {'tau': tau})
        sio.savemat('Output/l_b.mat', {'l_b': l_b})

    return {
        'realgdp_w': realgdp_w,
        'u_w': u_w,
        'u2_w': u2_w,
        'prod_w': prod_w,
        'phi_w': phi_w,
        'PDV_u_w': PDV_u_w,
        'PDV_u2_w': PDV_u2_w,
        'PDV_realgdp_w': PDV_realgdp_w,
        'migr_cell': migr_cell,
        'migr_ctry': migr_ctry,
        'l': l,
        'u': u,
        'u2': u2,
        'tau': tau,
        'realgdp': realgdp,
        'l_b': l_b,
        'vars': vars_for_run,
    }


if __name__ == '__main__':
    run_model()
