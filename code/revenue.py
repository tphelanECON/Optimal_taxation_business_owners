"""
Figures for revenue raised by taxes.
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

sig_grid = np.linspace(0.1, 0.2, 400)
lgrid = np.linspace(llow, 1-10**-6, 400)

c1,c2='lightsteelblue','darkblue'
def colorFader(c1,c2,mix):
    return mpl.colors.to_hex((1-mix)*np.array(mpl.colors.to_rgb(c1)) + mix*np.array(mpl.colors.to_rgb(c2)))

"""
Revenue raised from taxes for various values of gamma
"""

N=5
gamma_grid = np.linspace(1+1/N, 2, N)
fig, ax = plt.subplots()
for j in range(N):
    X = classes.CRRA(alpha=alpha, gamma=gamma_grid[j], rhoS=rhoS, rhoD=rhoD,
    sigma=sigma, umax=5, Nu=1000, mu_0=mu_0, mu_1=mu_1, llow=llow)
    color = colorFader(c1,c2,j/N)
    ax.plot(lgrid,100*X.revenue(X.sigma,lgrid),color=color,label="$\gamma$ = {0}".format(X.gamma),linewidth=1.5)
ax.set_xlabel('Leisure', fontsize=13)
ax.set_ylabel("Percent",fontsize=13)
plt.legend()
plt.title("Revenue raised per unit of wealth ($\sigma$ = {0})".format(X.sigma))
destin = '../main/figures/rev_lgrid.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig, ax = plt.subplots()
for j in range(N):
    X = classes.CRRA(alpha=alpha, gamma=gamma_grid[j], rhoS=rhoS, rhoD=rhoD,
    sigma=sigma, umax=5, Nu=1000, mu_0=mu_0, mu_1=mu_1, llow=llow)
    color = colorFader(c1,c2,j/N)
    ax.plot(sig_grid,100*X.revenue(sig_grid,X.llow),color=color,label="$\gamma$ = {0}".format(X.gamma),linewidth=1.5)
ax.set_xlabel('Volatility', fontsize=13)
ax.set_ylabel("Percent",fontsize=13)
plt.legend()
plt.title("Revenue raised per unit of wealth ($l$ = {0})".format(X.llow))
destin = '../main/figures/rev_sig_grid.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()
