"""
Example of optimal contract with logarithmic utility.

Relegated to appendix in 2023 version of paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import classes, parameters

sigma, alpha, llow = parameters.sigma, parameters.alpha, parameters.llow
mu_1, mu_0 = parameters.mu_1, parameters.mu_0
rhoS, rhoD = parameters.rhoS, parameters.rhoD

Y = classes.LOG(alpha=alpha, rhoS=rhoS, rhoD=rhoD, sigma=sigma,
umax=12, Nu=1200, mu_0=mu_0, mu_1=mu_1, llow=llow, nu=0.5)

"""
Compute value function and policy functions together with restricted-action functions.
"""

v = Y.solve_PFI()
cbar, l = Y.polupdate(v)
l_rest = Y.rest_l
c_rest = np.exp(Y.sigma**2*Y.E(Y.rest_l)**2/(2*Y.rho))*Y.rest_l**(-Y.alpha/(1-Y.alpha))

"""
Produce error estimates for HJB equation
"""

v_star = Y.rest_PA(Y.rest_l,Y.ugrid)
HJB_error = np.max(np.abs(Y.HJB_error(cbar,l,v)))
HJB_mean = np.mean(np.abs(Y.HJB_error(cbar,l,v)))
print("Maximum absolute error in HJB equation:", HJB_error)
print("Mean absolute error in HJB equation:", HJB_mean)

"""
Figures: leisure, consumption, risk-adjusted growth rate, and volatility.
"""

r_frac = 0.25

fig, ax = plt.subplots()
ax.plot(Y.ugrid,l,'b',label='Efficient',linewidth=1.5)
ax.plot(Y.ugrid,Y.rest_l,'b--',label='Restricted-action',linewidth=2.0)
plt.xlim([0,r_frac*Y.umax])
ax.legend()
ax.set_xlabel('Normalized utility', fontsize=13)
ax.set_title('Leisure', fontsize=13)
destin = '../main/figures/leisure_log.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig, ax = plt.subplots()
ax.plot(Y.ugrid,cbar,'b',label='Efficient', linewidth=1.5)
ax.plot(Y.ugrid,c_rest,'b--',label='Restricted-action',linewidth=2.0)
plt.xlim([0,r_frac*Y.umax])
plt.ylim([1.0, Y.rest(Y.llow,1)[0] + 0.04])
ax.legend(loc='lower left')
ax.set_xlabel('Normalized utility', fontsize=13)
ax.set_title('Consumption', fontsize=13)
destin = '../main/figures/consumption_log.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

bnds = Y.risk_adj(c_rest,Y.rest_l)[0] - 0.01, Y.risk_adj(c_rest,Y.rest_l)[int(Y.Nu/2)] + 0.01

fig, ax = plt.subplots()
ax.plot(Y.ugrid,Y.risk_adj(cbar,l),'b',label='Efficient',linewidth=1.5)
ax.plot(Y.ugrid,Y.risk_adj(c_rest,Y.rest_l),'b--',label='Restricted-action',linewidth=2.0)
ax.legend(loc='upper left')
plt.xlim([0,r_frac*Y.umax])
plt.ylim([bnds[0],bnds[-1]])
ax.set_xlabel('Normalized utility', fontsize=13)
ax.set_title('Risk-adjusted growth in utility', fontsize=13)
destin = '../main/figures/risk_adj_log.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

bnds = Y.sigma_u(c_rest,Y.rest_l)[0] + 0.01, Y.sigma_u(c_rest,Y.rest_l)[int(Y.Nu/2)] - 0.01

fig, ax = plt.subplots()
ax.plot(Y.ugrid,Y.sigma_u(cbar,l),'b',label='Efficient',linewidth=1.5)
ax.plot(Y.ugrid,Y.sigma_u(c_rest,Y.rest_l),'b--',label='Restricted-action',linewidth=2.0)
ax.legend(loc='lower left')
plt.xlim([0,r_frac*Y.umax])
plt.ylim([bnds[-1],bnds[0]])
ax.set_xlabel('Normalized utility', fontsize=13)
ax.set_title('Volatility', fontsize=13)
destin = '../main/figures/sig_u_log.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()
