import numpy as np
from math import gamma

def backward(H, T, vars):
    # Ensure global variables are available
    global a_norm, m2, tau0, pop0, trmult_reduced, earth_indices, H0, n, alpha, theta, Omega
    H0, a, a_norm, m2, _, tau0, pop0, _, _, _, _, _, trmult_reduced, n, earth_indices, _, _, _, _, _, _, alpha, theta, Omega = vars
    
    # Initialize parameters and output
    # Normalize population to population density
    popdens = np.copy(pop0)
    popdens[earth_indices] = popdens[earth_indices] / H0[earth_indices]
    popdens[np.isinf(popdens)] = 0
    popdens[np.isnan(popdens)] = 0

    # Parameter values
    lbar = 5.9174e+09
    lambda_ = 0.32
    gamma1 = 0.319
    gamma2 = 0.99246
    mu = 0.8
    nu = 0.15
    ksi = 125
    sigma = 4
    rad = 6371
    khi = lambda_ - (alpha - 1 + (lambda_ + gamma1 / ksi - (1 - mu)) * theta) / (1 + 2 * theta)
    kappa1 = ((mu * ksi + gamma1) / ksi) ** (-(mu + gamma1 / ksi) * theta) * \
             mu ** (mu * theta) * (ksi * nu / gamma1) ** (-gamma1 / ksi * theta) * \
             gamma(1 - (sigma - 1) / theta) ** (theta / (sigma - 1))

    # Initialize output variables
    l = np.zeros((n, T))
    u = np.zeros((n, T))
    w = np.zeros((n, T))
    phi = np.zeros((n, T))
    tau = np.zeros((n, T))
    realgdp = np.zeros((n, T))

    # 2. Simulate the model backwards

    # Initial guess for Lhat
    l_loop = np.copy(popdens[earth_indices])

    # Outer loop
    for t in range(T):
        print(f't={-t - 1}')

        # Next period's productivity
        if t > 0:
            taunext = tau[:, t - 1]
        else:
            taunext = tau0

        # Solve for Lhat
        error = 1e+10
        
        # Pre-computed quantities used in the while loop
        aa = a_norm ** (theta ** 2 / (1 + 2 * theta))
        aa2 = a_norm ** ((1 + theta) / (khi + Omega * (1 + theta) / (1 + 2 * theta) + theta / (1 + 2 * theta) * gamma1 / (ksi * gamma2)) * (1 + 2 * theta))
        exponent_l = (1 - lambda_ * theta + (1 + theta) / (1 + 2 * theta) * (alpha - 1 + (lambda_ + gamma1 / ksi - (1 - mu)) * theta))
        input_integral_outer = aa * H ** ((theta - theta ** 2 * Omega) / (1 + 2 * theta)) * \
                               taunext ** ((1 + theta) / (gamma2 * (1 + 2 * theta))) * \
                               m2 ** (-theta ** 2 / (1 + 2 * theta))
        input_integral_outer[np.isnan(input_integral_outer)] = 0
        input_l_inner = H ** (-(1 + Omega * (1 + theta)) / (khi + Omega * (1 + theta) / (1 + 2 * theta) + theta / (1 + 2 * theta) * gamma1 / (ksi * gamma2)) * (1 + 2 * theta)) * \
                        taunext ** (1 / (khi + Omega * (1 + theta) / (1 + 2 * theta) + theta / (1 + 2 * theta) * gamma1 / (ksi * gamma2)) * gamma2 * (1 + 2 * theta)) * \
                        m2 ** (-(1 + theta) / (khi + Omega * (1 + theta) / (1 + 2 * theta) + theta / (1 + 2 * theta) * gamma1 / (ksi * gamma2)) * (1 + 2 * theta))
        input_l_inner[H == 0] = 0
        
        # Inner loop - solve for l using equation (40)
        while error >= 1:
            l_old = np.copy(l_loop)
            input_integral_inner = input_integral_outer * \
                                   l_loop ** (exponent_l - Omega * theta ** 2 / (1 + 2 * theta) - theta * (1 + theta) / (1 + 2 * theta) * gamma1 / (ksi * gamma2))
            input_integral_inner[l_loop == 0] = 0
            
            # Matrix product
            rhs = np.dot(trmult_reduced, input_integral_inner)
            
            l_loop = aa2 * input_l_inner * rhs ** (1 / ((khi + Omega * (1 + theta) / (1 + 2 * theta) + theta / (1 + 2 * theta) * gamma1 / (ksi * gamma2)) * theta))
            error = np.sum((l_loop - l_old) ** 2)
        
        # Rescale L so that H * L sum to lbar
        l[:, t] = l_loop / np.sum(H * l_loop) * lbar
        
        # Back out productivity using equation (39)
        tau[:, t] = ((mu + gamma1 / ksi) / (gamma1 / ksi) * nu) ** (theta * gamma1 / (ksi * gamma2)) * \
                    taunext ** (1 / gamma2) * l[:, t] ** (-theta * gamma1 / (ksi * gamma2))
        avgprodtogamma2 = np.sum(tau[:, t]) / n
        tau[:, t] = avgprodtogamma2 ** (gamma2 - 1) * tau[:, t]
        
        # Calculate utility
        u[:, t] = m2 * l[:, t] ** Omega * (kappa1 ** (1 / Omega) * \
                  ((mu + gamma1 / ksi) / (gamma1 / ksi) * nu) ** (gamma1 / (ksi * gamma2)) * \
                  (np.sum(tau[:, t]) / n) ** (1 / theta * (1 - 1 / gamma2)) * \
                  (lbar / np.sum(H * l_loop)) ** (1 / theta - 2 * lambda_ + (alpha - 1 + (lambda_ + gamma1 / ksi - (1 - mu)) * theta) / theta - Omega - gamma1 / (ksi * gamma2)))
        
        # Calculate real GDP per capita using equation (22)
        realgdp[:, t] = u[:, t] / a_norm * l[:, t] ** lambda_
        
        # Calculate innovation using equation (12) and (13)
        phi[:, t] = (gamma1 / (nu * (gamma1 + mu * ksi))) ** (1 / ksi) * l[:, t] ** (1 / ksi)
        
        # Calculate wage using equation (23)
        w[:, t] = a_norm ** (-theta / (1 + 2 * theta)) * u[:, t] ** (theta / (1 + 2 * theta)) * H ** (-1 / (1 + 2 * theta)) * \
                  tau[:, t] ** (1 / (1 + 2 * theta)) * l[:, t] ** ((alpha - 1 + (lambda_ + gamma1 / ksi - (1 - mu)) * theta) / (1 + 2 * theta))
        
        # Normalize wages relative to Princeton, NJ (Python index adjustment)
        w[:, t] = w[:, t] / w[3198, t]  # Adjust index as necessary for your data

    # Handle NaN values
    realgdp[np.isnan(realgdp)] = 0
    tau[np.isnan(tau)] = 0
    phi[np.isnan(phi)] = 0
    w[np.isnan(w)] = 0
    u[np.isnan(u)] = 0
    l[np.isnan(l)] = 0

    return l, u, w, tau, phi, realgdp
