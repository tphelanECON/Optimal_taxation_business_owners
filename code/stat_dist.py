"""
Stationary distributions of normalized utility.
"""

import numpy as np
import matplotlib.pyplot as plt
import classes, parameters

sigma, alpha, llow = parameters.sigma, parameters.alpha, parameters.llow
mu_1, mu_0 = parameters.mu_1, parameters.mu_0
rhoS, rhoD = parameters.rhoS, parameters.rhoD

gamma = 2
gambar = (gamma-1)*(1 - alpha) + 1

X = classes.CRRA(alpha=alpha, gamma=gamma, rhoS=rhoS, rhoD=rhoD, sigma=sigma,
umax=5, Nu=1000, mu_0=mu_0, mu_1=mu_1, llow=llow)

inj = 150

v = X.solve_PFI()
cbar, l = X.polupdate(v)
l_rest = X.rest_l
c_rest = X.x(X.rest_l)**(1/(1-X.gammabar))*X.rest_l**(-X.alpha/(1-X.alpha))

first_lbar = np.where(l>X.llow)[0][0]
density = X.stat_dist(cbar,l,inj)
RA_density = X.stat_dist(c_rest,l_rest,inj)
ex_density = X.stat_dist(cbar[0] + 0*cbar, l[0] + 0*l, inj)

c1,c2='lightsteelblue','darkblue'
fig, ax = plt.subplots()
ax.plot(np.log(X.ugrid),np.log(RA_density),color=c1,label='Restricted-planner',linewidth=1.0)
ax.plot(np.log(X.ugrid),np.log(density),color=c2,label='Efficient',linewidth=1.0)
#ax.plot(np.log(X.ugrid),np.log(ex_density),color='k',label='highest restricted-action',linewidth=1.0)
plt.axvline(np.log(X.ugrid[inj]), color = 'k', label = 'initial value', linewidth=1.0)
plt.axvline(np.log(X.ugrid[first_lbar]), color = 'k', linestyle ='dotted', label = 'high effort',linewidth=1.0)
plt.xlim([np.log(X.ugrid[10]),np.log(r_frac*X.umax)])
plt.ylim([-20,max(np.log(density))])
ax.legend()
ax.set_xlabel('Log(Normalized utility)', fontsize=13)
ax.set_title('Log density', fontsize=13)
destin = '../main/figures/log_density_ex.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

c1,c2='lightsteelblue','darkblue'
fig, ax = plt.subplots()
ax.plot(np.log(X.ugrid),l_rest,color=c1,label='Restricted-planner',linewidth=1.0)
ax.plot(np.log(X.ugrid),l,color=c2,label='Efficient',linewidth=1.0)
plt.axvline(np.log(X.ugrid[inj]), color = 'k', label = 'initial value', linewidth=1.0)
plt.axvline(np.log(X.ugrid[first_lbar]), color = 'k', linestyle ='dotted', label = 'high effort',linewidth=1.0)
plt.xlim([np.log(X.ugrid[10]),np.log(r_frac*X.umax)])
ax.legend()
ax.set_xlabel('Log(Normalized utility)', fontsize=13)
ax.set_title('Policy functions', fontsize=13)
plt.show()


"""
Y = classes.LOG(alpha=alpha, rhoS=rhoS, rhoD=rhoD, sigma=sigma,
umax=12, Nu=1200, mu_0=mu_0, mu_1=mu_1, llow=llow, nu=0.5)

"""
