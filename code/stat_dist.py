"""
This script produces the following:

    * output and consumption per entrepreneur in stationary distribution as
    function of initial normalized utility of entrepreneurs.
    * calculation of efficient initial normalized utility for various values of
    labor share parameter beta and mass of entrepreneurs eta_E.
    * tails of distributions of consumption and productivity for each value, on
    a log-log scale, and comparison with restricted-action allocation associated
    with highest effort (lowest leisure).

I am only interested in upper tails so I plot values between 90th and 99.9th pct.
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

r_frac = 0.5
c1,c2='lightsteelblue','darkblue'
def colorFader(c1,c2,mix):
    return mpl.colors.to_hex((1-mix)*np.array(mpl.colors.to_rgb(c1)) + mix*np.array(mpl.colors.to_rgb(c2)))

v = X.solve_PFI()
cbar, l = X.polupdate(v)
l_rest = X.rest_l
c_rest = X.x(X.rest_l)**(1/(1-X.gammabar))*X.rest_l**(-X.alpha/(1-X.alpha))

"""
Compute aggregate output and consumption
"""

beta = 0.64
eta_list = [0.09,0.115,0.14]

M, C = {}, {}
u_norm, u, RC_excess = {}, {}, {}

M_rest, C_rest = {}, {}
u_norm_rest, u_rest, RC_excess_rest = {}, {}, {}
inj, l_init = {}, {}
tau_s, tau_a = {}, {}
revenue = {}

for i in range(len(eta_list)):
    eta_E = eta_list[i]
    u_norm[eta_E], u[eta_E], RC_excess[eta_E], (M[eta_E], C[eta_E]) = X.stat_u_norm(beta,eta_E,cbar,l)
    u_norm_rest[eta_E], u_rest[eta_E], RC_excess_rest[eta_E], (M_rest[eta_E], C_rest[eta_E]) = X.rest_stat_u_norm(beta,eta_E)
    inj[eta_E] = np.where(X.ugrid==u_norm_rest[eta_E])[0]
    l_init[eta_E] = l_rest[inj[eta_E]][0]
    tau_s[eta_E], tau_a[eta_E] = X.tax_signaling(l_init[eta_E])
    revenue[eta_E] = X.revenue(X.sigma,l_init[eta_E])

fig, ax = plt.subplots()
ax.plot(X.ugrid,M[eta_E],color=c2,label='Efficient',linewidth=2.0)
ax.plot(X.ugrid,M_rest[eta_E],color=c2,linestyle='--',label='Restricted-planner',linewidth=2.0)
ax.legend()
plt.xlim([0,0.25*X.umax])
plt.ylim([0.5,4.5])
ax.set_xlabel('Normalized utility', fontsize=13)
ax.set_title('Output per entrepreneur', fontsize=13)
destin = '../main/figures/M.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig, ax = plt.subplots()
ax.plot(X.ugrid,C[eta_E],color=c2,label='Efficient',linewidth=2.0)
ax.plot(X.ugrid,C_rest[eta_E],color=c2,linestyle='--',label='Restricted-planner',linewidth=2.0)
ax.legend(loc='upper left')
plt.xlim([0,0.25*X.umax])
plt.ylim([0,3])
ax.set_xlabel('Normalized utility', fontsize=13)
ax.set_title('Consumption per entrepreneur', fontsize=13)
destin = '../main/figures/C.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

"""
Distributions of consumption and firm size.
"""

c_grid, cdf_c, theta_grid, cdf_theta = {}, {}, {}, {}
cut = 10**-6

"""
Get tail parameters and initial values for lowest mass of entrepreneurs.
"""

tail_c, tail_theta = X.tail(X.sigma,X.llow)
tail_c_l, tail_theta_l = X.lower_tail(X.sigma,X.llow)
inj_final = np.where(np.exp(X.wgrid)>u_norm[eta_list[0]])[0][0]

inj_y = np.where(X.ygrid>0)[0][0]
cbar_w = interp1d(X.ugrid, cbar, fill_value="extrapolate")(np.exp(X.wgrid))
init_c = cbar_w[inj_final]*np.exp(X.wgrid[inj_final])*np.exp(X.ygrid[inj_y])
init_theta = np.exp(X.ygrid[inj_y])

inj_W = {}
for i in range(len(eta_list)):
    eta_E = eta_list[i]
    inj_W[eta_E] = np.where(np.exp(X.wgrid)>u_norm[eta_list[i]])[0][0]
    (c_grid[eta_E], cdf_c[eta_E]), (theta_grid[eta_E], cdf_theta[eta_E]) = X.marginal_dist(cbar, l, inj_W[eta_E], cut)

fig, ax = plt.subplots()
for i in range(len(eta_list)):
    eta_E = eta_list[i]
    ind = (cdf_c[eta_E] > 0.9)*(cdf_c[eta_E] < 0.999)
    ax.plot(np.log(c_grid[eta_E][ind]), np.log(1-cdf_c[eta_E][ind]), label="$\eta_E$ = {0}".format(eta_E), linewidth=1.5)
RA_cdf_c = X.RA_cdf(tail_c_l,tail_c,init_c,c_grid[eta_list[0]])
ind = (RA_cdf_c > 0.9)*(RA_cdf_c < 0.999)
ax.plot(np.log(c_grid[eta_list[0]][ind]), np.log(1-RA_cdf_c[ind]), label="Restrict-action", linewidth=1.5)
ax.legend()
ax.set_xlabel('log consumption', fontsize=13)
ax.set_title('log(1-CDF)', fontsize=13)
destin = '../main/figures/cdf_c_main.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig, ax = plt.subplots()
for i in range(len(eta_list)):
    eta_E = eta_list[i]
    ind = (cdf_theta[eta_E] > 0.9)*(cdf_theta[eta_E] < 0.999)
    ax.plot(np.log(theta_grid[eta_E][ind]), np.log(1-cdf_theta[eta_E][ind]), label="$\eta_E$ = {0}".format(eta_E), linewidth=1.5)
RA_cdf_theta = X.RA_cdf(tail_theta_l,tail_theta,init_theta,theta_grid[eta_list[0]])
ind = (RA_cdf_theta > 0.9)*(RA_cdf_theta < 0.999)
ax.plot(np.log(theta_grid[eta_list[0]][ind]), np.log(1-RA_cdf_theta[ind]), label="Restrict-action", linewidth=1.5)
ax.legend()
ax.set_xlabel('log productivity', fontsize=13)
ax.set_title('log(1-CDF)', fontsize=13)
destin = '../main/figures/cdf_theta_main.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

"""
CDFs
"""

fig, ax = plt.subplots()
for i in range(len(eta_list)):
    eta_E = eta_list[i]
    ind = (cdf_c[eta_E] > 0.75)*(cdf_c[eta_E] < 0.999)
    ax.plot(c_grid[eta_E], cdf_c[eta_E], label="$\eta_E$ = {0}".format(eta_E), linewidth=1.5)
RA_cdf_c = X.RA_cdf(tail_c_l,tail_c,init_c,c_grid[eta_list[0]])
ind = (RA_cdf_c > 0.75)*(RA_cdf_c < 0.999)
ax.plot(c_grid[eta_list[0]], RA_cdf_c, label="Restrict-action", linewidth=1.5)
plt.xlim([0,5])
ax.legend()
ax.set_xlabel('log consumption', fontsize=13)
ax.set_title('log(1-CDF)', fontsize=13)
#destin = '../main/figures/cdf_c_main.eps'
#plt.savefig(destin, format='eps', dpi=1000)
plt.show()
