"""
Example of optimal contract with CRRA utility. Placed in main text.

Note that if one reduces gamma, then one must also increase umax. The reason
for this is that
"""

import numpy as np
import matplotlib.pyplot as plt
import classes, parameters

sigma, alpha, llow = parameters.sigma, parameters.alpha, parameters.llow
mu_1, mu_0 = parameters.mu_1, parameters.mu_0
rhoS, rhoD = parameters.rhoS, parameters.rhoD

gamma = 2.
gambar = (gamma-1)*(1 - alpha) + 1

X = classes.CRRA(alpha=alpha, gamma=gamma, rhoS=rhoS, rhoD=rhoD, sigma=sigma,
umax=5, Nu=1000, mu_0=mu_0, mu_1=mu_1, llow=llow)

"""
Compute value function and policy functions together with restricted-action functions.
"""

v = X.solve_PFI()
cbar, l = X.polupdate(v)
l_rest = X.rest_l
c_rest = X.x(X.rest_l)**(1/(1-X.gammabar))*X.rest_l**(-X.alpha/(1-X.alpha))

"""
Produce error estimates for HJB equation
"""

v_star = X.rest_PA(X.rest_l,X.ugrid)
HJB_error = np.max(np.abs(X.HJB_error(cbar,l,v)))
HJB_mean = np.mean(np.abs(X.HJB_error(cbar,l,v)))
print("Maximum absolute error in HJB equation:", HJB_error)
print("Mean absolute error in HJB equation:", HJB_mean)
check = X.check_concave(v) < 0
check_star = X.check_concave(v_star) < 0
np.mean(check)

"""
Figures: leisure, consumption, risk-adjusted growth rate, and volatility.

Value function is computed but not saved here as it is not shown in the main text.
"""

r_frac = 0.5

"""
fig, ax = plt.subplots()
ax.plot(X.ugrid,v,'b',label='Efficient',linewidth=1.5)
ax.plot(X.ugrid,X.rest(X.llow,X.ugrid)[1],'b--',label='Highest effort',linewidth=1.5)
ax.plot(X.ugrid,X.rest(X.lhigh,X.ugrid)[1],'b--',label='Lowest effort',linewidth=1.5)
ax.plot(X.ugrid,X.rest(X.lhigh-10**-6,X.ugrid)[1],'b--',label='Lowest non-zero effort',linewidth=1.5)
#ax.plot(X.ugrid,X.rest_l,'b--',label='Restricted-action',linewidth=1.0)
plt.xlim([0,r_frac*X.umax])
ax.legend()
ax.set_xlabel('Normalized utility', fontsize=13)
ax.set_title('Value function', fontsize=13)
#destin = '../main/figures/leisure_ex.eps'
#plt.savefig(destin, format='eps', dpi=1000)
plt.show()
"""

fig, ax = plt.subplots()
ax.plot(X.ugrid,l,'b',label='Efficient',linewidth=1.5)
ax.plot(X.ugrid,X.rest_l,'b--',label='Restricted-action',linewidth=1.0)
plt.xlim([0,r_frac*X.umax])
ax.legend()
ax.set_xlabel('Normalized utility', fontsize=13)
ax.set_title('Leisure', fontsize=13)
destin = '../main/figures/leisure_ex.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig, ax = plt.subplots()
ax.plot(X.ugrid,cbar,'b',label='Efficient', linewidth=1.5)
ax.plot(X.ugrid,c_rest,'b--',label='Restricted-action',linewidth=1.0)
plt.xlim([0,r_frac*X.umax])
#plt.xlim([0,3])
plt.ylim([1.0, X.rest(X.llow,1)[0] + 0.15])
ax.legend()
ax.set_xlabel('Normalized utility', fontsize=13)
ax.set_title('Consumption coefficient', fontsize=13)
destin = '../main/figures/consumption_ex.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

bnds = X.risk_adj(c_rest,X.rest_l)[0] - 0.01, X.risk_adj(c_rest,X.rest_l)[int(X.Nu/2)] + 0.01

fig, ax = plt.subplots()
ax.plot(X.ugrid,X.risk_adj(cbar,l),'b',label='Efficient',linewidth=1.5)
ax.plot(X.ugrid,X.risk_adj(c_rest,X.rest_l),'b--',label='Restricted-action',linewidth=1.0)
ax.legend(loc='upper left')
plt.xlim([0,r_frac*X.umax])
plt.ylim([bnds[0],bnds[-1]])
ax.set_xlabel('Normalized utility', fontsize=13)
ax.set_title('Risk-adjusted growth in utility', fontsize=13)
destin = '../main/figures/risk_adj_ex.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

bnds = X.sigma_u(c_rest,X.rest_l)[0] + 0.01, X.sigma_u(c_rest,X.rest_l)[int(X.Nu/2)] - 0.01

fig, ax = plt.subplots()
ax.plot(X.ugrid,X.sigma_u(cbar,l),'b',label='Efficient',linewidth=1.5)
ax.plot(X.ugrid,X.sigma_u(c_rest,X.rest_l),'b--',label='Restricted-action',linewidth=1.0)
ax.legend(loc='upper right')
#plt.xlim([0,r_frac*X.umax])
plt.ylim([bnds[-1],bnds[0]])
ax.set_xlabel('Normalized utility', fontsize=13)
ax.set_title('Volatility', fontsize=13)
destin = '../main/figures/sig_u_ex.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()
