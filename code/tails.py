"""
Figures for the Pareto exponents of consumption and firm size
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

Y = classes.LOG(alpha=alpha, rhoS=rhoS, rhoD=rhoD, sigma=sigma,
umax=12, Nu=1200, mu_0=mu_0, mu_1=mu_1, llow=llow, nu=0.5)

lgrid = np.linspace(X.llow, 1-10**-6, 400)
sig_grid = np.linspace(X.sigma/2, X.sigma, 400)

"""
Tail parameters for consumption and firm size
"""

"""
CRRA with gamma = 2
"""

fig, ax1 = plt.subplots()
lns1 = ax1.plot(lgrid, X.tail(X.sigma,lgrid)[0],'b',label='Consumption (LHS)',linewidth=1.5)
ax2=ax1.twinx()
lns2 = ax2.plot(lgrid, X.tail(X.sigma,lgrid)[1],'k',label='Firm size (RHS)',linewidth=1.5)
leg = lns1 + lns2
labs = [l.get_label() for l in leg]
ax1.legend(leg, labs, loc=0)
ax1.set_xlabel('Leisure', fontsize=13)
plt.title("Pareto exponents ($\sigma$ = {0})".format(Y.sigma))
destin = '../main/figures/tails.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig, ax1 = plt.subplots()
lns1 = ax1.plot(sig_grid, X.tail(sig_grid, X.llow)[0],'b',label='Consumption (LHS)',linewidth=1.5)
ax2=ax1.twinx()
lns2 = ax2.plot(sig_grid, X.tail(sig_grid, X.llow)[1],'k',label='Firm size (RHS)',linewidth=1.5)
leg = lns1 + lns2
labs = [l.get_label() for l in leg]
ax1.legend(leg, labs, loc=0)
ax1.set_xlabel('Volatility', fontsize=13)
destin = '../main/figures/tails_sig.eps'
plt.title("Pareto exponents ($l$ = {0})".format(X.llow))
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

"""
Logarithmic utility (placed in the appendix)
"""

fig, ax1 = plt.subplots()
lns1 = ax1.plot(lgrid, Y.tail(Y.sigma,lgrid)[0],'b',label='Consumption (LHS)',linewidth=1.5)
ax2=ax1.twinx()
lns2 = ax2.plot(lgrid, Y.tail(Y.sigma,lgrid)[1],'k',label='Firm size (RHS)',linewidth=1.5)
leg = lns1 + lns2
labs = [l.get_label() for l in leg]
ax1.legend(leg, labs, loc=0)
ax1.set_xlabel('Leisure', fontsize=13)
plt.title("Pareto exponents ($\sigma$ = {0})".format(Y.sigma))
destin = '../main/figures/tails_log.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()

fig, ax1 = plt.subplots()
lns1 = ax1.plot(sig_grid, Y.tail(sig_grid, Y.llow)[0],'b',label='Consumption (LHS)',linewidth=1.5)
ax2=ax1.twinx()
lns2 = ax2.plot(sig_grid, Y.tail(sig_grid, Y.llow)[1],'k',label='Firm size (RHS)',linewidth=1.5)
leg = lns1 + lns2
labs = [l.get_label() for l in leg]
ax1.legend(leg, labs, loc=0)
ax1.set_xlabel('Volatility', fontsize=13)
plt.title("Pareto exponents ($l$ = {0})".format(Y.llow))
destin = '../main/figures/tails_sig_log.eps'
plt.savefig(destin, format='eps', dpi=1000)
plt.show()
