import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.spatial.distance import cosine
import statsmodels.api as sm

# Parameters
alpha = 0.06
lambda_ = 0.32
gamma1 = 0.319
mu = 0.8
ksi = 125
theta = 6.5
rad = 6371
psi = 1.8

# Load data
H0 = loadmat('H0.mat')['H0']
earth_indices = np.where(H0 > 0)[0]
n = len(earth_indices)
H = H0.flatten()[earth_indices]
C = pd.read_csv('C.csv', header=None).values.flatten()
Ctry = C[earth_indices]

# Compute distances
print('Calculate distances...')
distmat = np.zeros((180, 360, 180, 360))
for i in range(180):
    for j in range(360):
        if H0[i, j] > 0:
            for k in range(180):
                for l in range(360):
                    if H0[k, l] > 0:
                        lat1 = i / 180 * np.pi
                        lat2 = k / 180 * np.pi
                        long1 = j / 180 * np.pi
                        long2 = l / 180 * np.pi
                        distmat[i, j, k, l] = rad * np.real(
                            np.arccos(np.sin(lat1) * np.sin(lat2) +
                                      np.cos(long1 - long2) * np.cos(lat1) * np.cos(lat2))
                        )

distmat = distmat.reshape(180 * 360, 180 * 360)
distmat_reduced = distmat[earth_indices, :][:, earth_indices]

# Trade flows, GDP
trademat_reduced = np.zeros((n, n))
gdp1_reduced = np.zeros((n, n))
gdp2_reduced = np.zeros((n, n))
pop1_reduced = np.zeros((n, n))
pop2_reduced = np.zeros((n, n))
cell1_reduced = np.zeros((n, n))
cell2_reduced = np.zeros((n, n))
ctry1_reduced = np.zeros((n, n))
ctry2_reduced = np.zeros((n, n))

a_H0 = loadmat('a_H0.mat')['a_H0']
a = a_H0
ubar = pd.read_csv('ubar.csv', header=None).values.flatten()
ubar[np.isnan(ubar)] = 0
ubar[np.isinf(ubar)] = 0
u0 = np.exp(psi * ubar[earth_indices])
a_norm = a * u0

l = loadmat('Output/NF_1000/l.mat')['l']
tau = loadmat('Output/NF_1000/tau.mat')['tau']
u = loadmat('Output/NF_1000/u.mat')['u']
trmult_reduced = loadmat('trmult_reduced.mat')['trmult_reduced']

w = (a_norm**(-theta / (1 + 2 * theta)) *
     u[:, 0]**(theta / (1 + 2 * theta)) *
     H**(-1 / (1 + 2 * theta)) *
     tau[:, 0]**(1 / (1 + 2 * theta)) *
     l[:, 0]**((alpha - 1 + (lambda_ + gamma1 / ksi - (1 - mu)) * theta) / (1 + 2 * theta))
    )

trsharesum = trmult_reduced @ (tau[:, 0] * l[:, 0]**(alpha - (1 - mu - gamma1 / ksi) * theta) * w**(-theta))

print('Calculate trade flows...')
for i in range(n):
    for j in range(n):
        if i != j:
            trademat_reduced[i, j] = (
                tau[j, 0] * l[j, 0]**(alpha - (1 - mu - gamma1 / ksi) * theta) *
                w[j]**(-theta) * trmult_reduced[i, j] / trsharesum[i] *
                w[i] * H[i] * l[i, 0]
            )
            gdp1_reduced[i, j] = w[i] * H[i] * l[i, 0]
            gdp2_reduced[i, j] = w[j] * H[j] * l[j, 0]
            pop1_reduced[i, j] = H[i] * l[i, 0]
            pop2_reduced[i, j] = H[j] * l[j, 0]
            cell1_reduced[i, j] = i
            cell2_reduced[i, j] = j
            ctry1_reduced[i, j] = Ctry[i]
            ctry2_reduced[i, j] = Ctry[j]

# Save results
np.savez('gravity.npz', 
         trademat_reduced=trademat_reduced,
         gdp1_reduced=gdp1_reduced,
         gdp2_reduced=gdp2_reduced,
         pop1_reduced=pop1_reduced,
         pop2_reduced=pop2_reduced,
         cell1_reduced=cell1_reduced,
         cell2_reduced=cell2_reduced,
         ctry1_reduced=ctry1_reduced,
         ctry2_reduced=ctry2_reduced,
         distmat_reduced=distmat_reduced)

# Load results
data = np.load('gravity.npz')
trademat_reduced = data['trademat_reduced']
gdp1_reduced = data['gdp1_reduced']
gdp2_reduced = data['gdp2_reduced']
pop1_reduced = data['pop1_reduced']
pop2_reduced = data['pop2_reduced']
cell1_reduced = data['cell1_reduced']
cell2_reduced = data['cell2_reduced']
ctry1_reduced = data['ctry1_reduced']
ctry2_reduced = data['ctry2_reduced']
distmat_reduced = data['distmat_reduced']

# Regress log trade on log distance
distv = distmat_reduced.flatten()
tradev = trademat_reduced.flatten()
gdp1v = gdp1_reduced.flatten()
gdp2v = gdp2_reduced.flatten()
pop1v = pop1_reduced.flatten()
pop2v = pop2_reduced.flatten()
cell1v = cell1_reduced.flatten()
cell2v = cell2_reduced.flatten()
ctry1v = ctry1_reduced.flatten()
ctry2v = ctry2_reduced.flatten()

# Remove zero values where countries are not different
zro = np.sign(distv * tradev * gdp1v * gdp2v * pop1v * pop2v)
for i in range(len(ctry1v)):
    if ctry1v[i] != ctry2v[i]:
        zro[i] = 0

mask = zro != 0
distv = distv[mask]
tradev = tradev[mask]
gdp1v = gdp1v[mask]
gdp2v = gdp2v[mask]
pop1v = pop1v[mask]
pop2v = pop2v[mask]

# Prepare the data for regression
X = np.column_stack([np.ones(len(tradev)), np.log(gdp1v), np.log(gdp2v), np.log(pop1v), np.log(pop2v), np.log(distv)])
y = np.log(tradev)

# Perform regression
model = sm.OLS(y, X)
results = model.fit()

print(results.summary())
