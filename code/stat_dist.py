"""
Stationary distributions of normalized utility.

Compute the resource constraints (LHS and RHS) for both restricted-action
and efficient allocations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import classes, parameters

sigma, alpha, llow = parameters.sigma, parameters.alpha, parameters.llow
mu_1, mu_0 = parameters.mu_1, parameters.mu_0
rhoS, rhoD = parameters.rhoS, parameters.rhoD

gamma = 2
gambar = (gamma-1)*(1 - alpha) + 1

X = classes.CRRA(alpha=alpha, gamma=gamma, rhoS=rhoS, rhoD=rhoD, sigma=sigma,
umax=5, Nu=1000, mu_0=mu_0, mu_1=mu_1, llow=llow)

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
eta_list = [0.076,0.115]

M, C = {}, {}
y, u, RC_excess = {}, {}, {}

M_rest, C_rest = {}, {}
y_rest, u_rest, RC_excess_rest = {}, {}, {}

for i in range(len(eta_list)):
    eta_E = eta_list[i]
    y[eta_E], u[eta_E], RC_excess[eta_E], (M[eta_E], C[eta_E]) = X.stat_y(beta,eta_E,cbar,l)
    y_rest[eta_E], u_rest[eta_E], RC_excess_rest[eta_E], (M_rest[eta_E], C_rest[eta_E]) = X.rest_stat_y(beta,eta_E)

for i in range(len(eta_list)):
    fig, ax = plt.subplots()
    eta_E = eta_list[i]
    ax.plot(X.ugrid,eta_E*M[eta_E]/(1-beta),color=c2,label='Output',linewidth=2.0)
    ax.plot(X.ugrid,eta_E*M_rest[eta_E]/(1-beta),color=c2,linestyle='--',linewidth=2.0)
    ax.plot(X.ugrid,eta_E*C[eta_E] + (1-eta_E)*X.ugrid,color=c1,label='Consumption',linewidth=2.0)
    ax.plot(X.ugrid,eta_E*C_rest[eta_E] + (1-eta_E)*X.ugrid,color=c1,linestyle='--',linewidth=2.0)
    plt.xlim([0,0.4*X.umax])
    plt.ylim([0,2.2])
    ax.legend(loc='upper left')
    ax.set_xlabel('Normalized utility', fontsize=13)
    ax.set_title('Entrepreneurs {0} percent of population'.format(100*eta_E), fontsize=13)
    destin = '../main/figures/RC_{0}.eps'.format(10**3*eta_E)
    plt.savefig(destin, format='eps', dpi=1000)
    plt.show()

"""
Plot density of normalized utility
"""

inj = np.where(X.ugrid==y_rest[eta_list[0]])[0]

first_lbar = np.where(l>X.llow)[0][0]
density = X.stat_dist(cbar,l,inj)
RA_density = X.stat_dist(c_rest,l_rest,inj)
ex_density = X.stat_dist(cbar[0] + 0*cbar, l[0] + 0*l, inj)

fig, ax = plt.subplots()
ax.plot(np.log(X.ugrid),np.log(RA_density),color=c1,label='Restricted-planner',linewidth=2.0)
ax.plot(np.log(X.ugrid),np.log(density),color=c2,label='Efficient',linewidth=2.0)
plt.axvline(np.log(X.ugrid[inj]), color = 'k', label = 'initial value', linewidth=2.0)
plt.axvline(np.log(X.ugrid[first_lbar]), color = 'k', linestyle ='dotted', label = 'high effort',linewidth=2.0)
plt.xlim([np.log(X.ugrid[10]),np.log(r_frac*X.umax)])
plt.ylim([-20,max(np.log(density))])
ax.legend()
ax.set_xlabel('Log(Normalized utility)', fontsize=13)
ax.set_title('Log density', fontsize=13)
destin = '../main/figures/log_density_ex.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig, ax = plt.subplots()
ax.plot(np.log(X.ugrid),l_rest,color=c1,label='Restricted-planner',linewidth=2.0)
ax.plot(np.log(X.ugrid),l,color=c2,label='Efficient',linewidth=2.0)
plt.axvline(np.log(X.ugrid[inj]), color = 'k', label = 'initial value', linewidth=2.0)
plt.axvline(np.log(X.ugrid[first_lbar]), color = 'k', linestyle ='dotted', label = 'high effort',linewidth=2.0)
plt.xlim([np.log(X.ugrid[10]),np.log(r_frac*X.umax)])
ax.legend()
ax.set_xlabel('Log(Normalized utility)', fontsize=13)
ax.set_title('Policy functions', fontsize=13)
plt.show()
