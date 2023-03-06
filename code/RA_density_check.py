"""
This script computes the density of the restricted-action allocations for the
highest effort (lowest leisure) and compares it with the closed-form expression.

The purpose is to provide confidence in the (somewhat non-standard) non-local
construction I need to use to deal with high correlation in the numerical method.

I compare the cdfs with one another globally and check that they coincide.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import classes, parameters
from scipy.interpolate import interp1d

sigma, alpha, llow = parameters.sigma, parameters.alpha, parameters.llow
mu_1, mu_0 = parameters.mu_1, parameters.mu_0
rhoS, rhoD = parameters.rhoS, parameters.rhoD

gamma = 2
gambar = (gamma-1)*(1 - alpha) + 1

X = classes.CRRA(alpha=alpha, gamma=gamma, rhoS=rhoS, rhoD=rhoD, sigma=sigma,
umax=10, Nu=1600, Ny=800, mu_0=mu_0, mu_1=mu_1, llow=llow, mbar = 4)

l_fix = X.llow
c_fix = X.c_rest[0]
inj = np.where(np.exp(X.wgrid)>1)[0][0]
cut = 10**-6

c_check, l_check = 0*X.ugrid + c_fix, 0*X.ugrid + l_fix
(c_grid, cdf_c), (theta_grid, cdf_theta) = X.marginal_dist(c_check, l_check, inj,cut)

"""
Create the restricted-action distributions.
"""

tail_c, tail_theta = X.tail(X.sigma,l_fix)
tail_c_l, tail_theta_l = X.lower_tail(X.sigma,l_fix)

inj_y = np.where(X.ygrid>0)[0][0]
cbar_w = interp1d(X.ugrid, c_check, fill_value="extrapolate")(np.exp(X.wgrid))
init_c = cbar_w[inj]*np.exp(X.wgrid[inj])*np.exp(X.ygrid[inj_y])
init_theta = np.exp(X.ygrid[inj_y])

RA_cdf_c = X.RA_cdf(tail_c_l,tail_c,init_c,c_grid)
RA_cdf_theta = X.RA_cdf(tail_theta_l,tail_theta,init_theta,theta_grid)

fig, ax = plt.subplots()
ax.plot(c_grid[cdf_c < 0.99], cdf_c[cdf_c < 0.99], 'b', label='Computed', linewidth=1.5)
ax.plot(c_grid[RA_cdf_c < 0.99], RA_cdf_c[RA_cdf_c < 0.99], 'b--', label='Closed-form', linewidth=1.5)
ax.legend()
ax.set_xlabel('Consumption', fontsize=13)
ax.set_title('CDF', fontsize=13)
destin = '../main/figures/cdf_c_check.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig, ax = plt.subplots()
ax.plot(theta_grid[cdf_theta < 0.99], cdf_theta[cdf_theta < 0.99], 'b', label='Computed', linewidth=1.5)
ax.plot(theta_grid[RA_cdf_theta < 0.99], RA_cdf_theta[RA_cdf_theta < 0.99], 'b--', label='Closed-form', linewidth=1.5)
ax.legend()
ax.set_xlabel('Productivity', fontsize=13)
ax.set_title('CDF', fontsize=13)
destin = '../main/figures/cdf_theta_check.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()
