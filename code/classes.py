"""
Class constructors for "The Optimal Taxation of Business Owners".

Author: Thomas Phelan.
Email: tom.phelan@clev.frb.org.

Two class constructors: CRRA and LOG (treated separately to avoid confusion)

Convention for grids: Nu + 1 points if we include boundaries.
"""

import numpy as np
import scipy.sparse as sp
import scipy.optimize as scopt
from scipy.sparse import linalg
from scipy.optimize import minimize
from scipy.optimize import Bounds
import time

class CRRA(object):
    def __init__(self, alpha=0.33, gamma=3.0, rhoS=0.05, rhoD=0.05, sigma=0.3,
    mu_0 = 0.08, mu_1 = 0., umax = 6, Nu = 200, llow = 1/3, scrap = 0,
    maxiter = 20, tol = 10**-5, Dt = 10**(-6)):
        self.alpha, self.gamma, self.gammabar = alpha, gamma, 1 - (1-gamma)*(1-alpha)
        self.rhoS, self.rhoD, self.rho = rhoS, rhoD, rhoS + rhoD
        self.mu_1, self.mu_0, self.sigma = mu_1, mu_0, sigma
        self.umax, self.llow, self.lhigh = umax, llow, 1
        self.Deltau, self.Nu, self.Dt = (umax - 0)/Nu, Nu, Dt
        self.scrap, self.maxiter, self.tol = scrap, maxiter, tol
        self.ugrid = np.linspace(self.Deltau, self.umax - self.Deltau, self.Nu-1)
        self.rest_l = np.asarray([self.rest_pol(self.ugrid[i]) for i in range(self.Nu-1)])
        self.v_star = self.rest_PA(self.rest_l,self.ugrid)
        self.global_suff = self.rho < (1/self.alpha-1)*(self.mu_0 - self.mu_1)*self.llow \
        + self.sigma**2*(2*self.gammabar-1)*(self.gammabar-1)/2
        if self.global_suff==True:
            print("Normalized utility falls with positive shocks for highest effort")
        self.finite = self.rhoD > self.mu_theta(self.llow)
        if self.finite==False:
            print("PROBLEM: output diverges for highest effort")
        self.c_bnd = 0.5, 1.5*(self.rest(self.llow,1)[0])
        self.x0 = ((self.c_bnd[0] + self.c_bnd[1])/2, (self.llow + 1.0)/2)

    #"skin-in-the-game" function defined in the principal-agent setting
    def E(self,l):
        return (l<1)*self.rho*self.alpha/((1-self.alpha)*(self.mu_0 - self.mu_1)*l)

    #mean growth in productivity as function of leisure
    def mu_theta(self,l):
        return (self.mu_0 - (self.mu_0-self.mu_1)*l)

    #volatility of productivity growth. Ceases upon "retirement"
    def sigma_theta(self,l):
        return self.sigma*(l<1)

    #mean of "normalized utility"
    def mu_u(self,cbar,l):
        x = (cbar**(1-self.alpha)*l**self.alpha)**(1-self.gamma)
        return self.rho*(1 - x)/(1-self.gammabar) \
        + (self.gammabar-1)*self.sigma**2*self.E(l)**2*x**2/2 \
        + (self.sigma*self.E(l)*x-self.sigma_theta(l))**2/2 \
        - self.mu_theta(l) + self.sigma_theta(l)**2/2

    #volatility of "normalized utility"
    def sigma_u(self,cbar,l):
        x = (cbar**(1-self.alpha)*l**self.alpha)**(1-self.gamma)
        return self.sigma*(self.E(l)*x - 1)*(l<1)

    def risk_adj(self,cbar,l):
        return self.mu_u(cbar,l) - self.sigma_u(cbar,l)**2/2 - (self.mu_theta(l) - self.sigma_theta(l)**2/2)

    def rest(self,l,u):
        ind = l<1
        c_rest = self.x(l)**(1/(1-self.gammabar))*l**(-self.alpha/(1-self.alpha))*u
        v_rest = (ind + (1-ind)*self.scrap)*(self.rho - self.mu_theta(l))**(-1) \
        - ((2*self.gammabar-1)/(2*self.gammabar - self.x(l)))*c_rest/self.rho
        slope = ((2*self.gammabar-1)/(2*self.gammabar - self.x(l)))*self.rho**(-1)*self.x(l)**(1/(1-self.gammabar))*l**(-self.alpha/(1-self.alpha))
        return c_rest, v_rest, slope

    def rest_PA(self,l,u):
        ind = l<1
        return (ind + (1-ind)*self.scrap)*(self.rho - self.mu_theta(l))**(-1) - self.rest(l,1)[2]*u

    def rest_pol(self,u):
        Y = -self.rest_PA(np.linspace(self.llow, 1, 500),u)
        return np.linspace(self.llow, 1, 500)[Y.argmin()]

    def x(self,l):
        with np.errstate(divide='ignore',invalid='ignore'):
            A = self.rho**(-1)*self.E(l)**2*(self.gammabar-1)*(self.gammabar-1/2)*self.sigma**2
            x = (-1 + np.sqrt(1 + 4*A))/(2*A)
        return np.nan_to_num(x,nan=1)

    def tran_func(self,cbar,l):
        tran_func = {}
        mu_u, sig_u = self.mu_u(cbar,l), self.sigma_u(cbar,l)
        mu_theta, sig_theta = self.mu_theta(l), self.sigma_theta(l)
        x = (cbar**(1-self.alpha)*l**self.alpha)**(1-self.gamma)
        mu_hat1 = self.rho*x/(self.gammabar - 1) + self.gammabar*self.sigma**2*self.E(l)**2*x**2/2
        mu_hat2 = self.rho/(self.gammabar - 1) + self.mu_theta(l)
        u, du = self.ugrid, self.Deltau
        tran_func[(1)] = mu_hat1*u/du + sig_u**2*u**2/(2*du**2)
        tran_func[(0)] = -self.rho + mu_theta - (mu_hat1 + mu_hat2)*u/du - sig_u**2*u**2/du**2
        tran_func[(-1)] = mu_hat2*u/du + sig_u**2*u**2/(2*du**2)
        return tran_func

    def T_tran(self,cbar,l):
        #diagonal:
        row = np.arange(self.Nu-1)
        diag = self.tran_func(cbar,l)[(0)]
        T = self.T_func(row,row,diag)
        #up:
        row = np.arange(self.Nu-2)
        diag = self.tran_func(cbar,l)[(1)][:-1]
        T = T + self.T_func(row,row+1,diag)
        #down:
        row = np.arange(1,self.Nu-1)
        diag = self.tran_func(cbar,l)[(-1)][1:]
        T = T + self.T_func(row,row-1,diag)
        return T

    def v(self,cbar,l):
        ind = l<1
        b = (ind + (1-ind)*self.scrap) - cbar*self.ugrid
        b[-1] = b[-1] + self.tran_func(cbar,l)[(1)][-1]*self.rest_PA(1,self.umax)
        b[0] = b[0] + self.tran_func(cbar,l)[(-1)][0]*self.rest_PA(self.llow,0)
        return sp.linalg.spsolve(-self.T_tran(cbar,l), b)

    def polupdate(self,v):
        cbar, l, cl1 = np.zeros(self.Nu-1,), np.zeros(self.Nu-1,), np.zeros(self.Nu-1,)
        eval, eval_bnd = np.zeros(self.Nu-1,), np.zeros(self.Nu-1,)
        u, vbig = self.ugrid, np.zeros(self.Nu+1,)
        vbig[1:-1] = v
        vbig[0], vbig[-1] = self.rest_PA(self.llow,0), self.rest_PA(1,self.umax)
        vF, vB = (vbig[2:] - vbig[1:-1])/self.Deltau, (vbig[1:-1] - vbig[0:-2])/self.Deltau
        vC2 = (vbig[2:] - 2*vbig[1:-1] + vbig[0:-2])/self.Deltau**2
        bounds = Bounds([self.c_bnd[0], self.llow], [self.c_bnd[1], 1.0])
        x0 = [(self.c_bnd[0] + self.c_bnd[1])/2, (self.llow + 1.0)/2]
        for i in range(self.Nu-1):
            f = lambda x: -self.M(x[0], x[1], u[i], v[i], vF[i], vB[i], vC2[i])
            f_jac = lambda x: -self.M_jac(x[0], x[1], u[i], v[i], vF[i], vB[i], vC2[i])
            res = minimize(f, x0, bounds=bounds,tol=10**-9) #jac=f_jac,
            cbar[i], l[i] = res.x[0], res.x[1]
            f_bnd = lambda c: -self.M_bnd(c, u[i], v[i], vF[i], vB[i], vC2[i])
            cl1[i] = (np.maximum(-self.rho*vF[i], self.c_bnd[0]**self.gammabar))**(1/self.gammabar)
            eval[i] = self.M(cbar[i], l[i], u[i], v[i], vF[i], vB[i], vC2[i])
            eval_bnd[i] = self.M_bnd(cl1[i], u[i], v[i], vF[i], vB[i], vC2[i])
            ind_bnd = eval[i] < eval_bnd[i]
            cbar[i] = cbar[i]*(1-ind_bnd) + ind_bnd*cl1[i]
            l[i] = l[i]*(1-ind_bnd) + ind_bnd
            x0 = [cbar[i], l[i]]
        return cbar, l

    def solve_PFI(self):
        v, i, eps = self.v_star, 1, 1
        while i < self.maxiter and eps > self.tol:
            tic = time.time()
            cbar, l = self.polupdate(v)
            toc = time.time()
            print("Time taken:", toc-tic)
            v1 = self.v(cbar,l)
            eps = np.sum(np.abs(v1-v))/self.Nu
            v, i = v1, i+1
            print("Difference in PFI:", eps, "Iterations:", i-1)
        return v

    def HJB_error(self,cbar,l,v):
        ind = l<1
        b = (ind + (1-ind)*self.scrap) - cbar*self.ugrid
        b[-1] = b[-1] + self.tran_func(cbar,l)[(1)][-1]*self.rest_PA(1,self.umax)
        b[0] = b[0] + self.tran_func(cbar,l)[(-1)][0]*self.rest_PA(self.llow,0)
        return b + self.T_tran(cbar,l)*v

    def M(self, cbar, l, u, v, vF, vB, vC2):
        mu_u, sig_u = self.mu_u(cbar,l), self.sigma_u(cbar,l)
        mu_theta, sig_theta = self.mu_theta(l), self.sigma_theta(l)
        mu_hat1 = self.rho*(cbar**(1-self.alpha)*l**self.alpha)**(1-self.gamma)/(self.gammabar - 1) \
        + self.gammabar*self.sigma**2*self.E(l)**2*(cbar**(1-self.alpha)*l**self.alpha)**(2-2*self.gamma)/2
        mu_hat2 = self.rho/(self.gammabar - 1) + self.mu_theta(l)
        ind = l<1
        return (ind + (1-ind)*self.scrap) - cbar*u + mu_theta*v + sig_u**2*u**2*vC2/2 + (mu_hat1*vF - mu_hat2*vB)*u

    #following uses
    # (d/dcbar)x = (1-self.gammabar)*x/cbar, (d/dl)x = self.alpha*(1-self.gamma)*x/l
    # (d/dl)E(l) = -E(l)/l
    # (d/dl)(E(l)x(l)) = E'(l)x(l) + E(l)x'(l) = -E(l)x(l)/l + self.alpha*(1-self.gamma)*E(l)x(l)/l = (self.alpha*(1-self.gamma)-1)E(l)x(l)/l
    def M_jac(self, cbar, l, u, v, vF, vB, vC2):
        x = (cbar**(1-self.alpha)*l**self.alpha)**(1-self.gamma)
        Mc = -u + (-self.rho*x/cbar + self.gammabar*self.sigma**2*self.E(l)**2*(1-self.gammabar)*x**2/cbar)*u*vF \
         + self.sigma**2*(self.E(l)*x - 1)*(1-self.gammabar)*self.E(l)*x*u**2*vC2/cbar
        Ml = -(self.mu_0 - self.mu_1)*(v - u*vB) + (-self.rho*self.alpha*x/(1-self.alpha) \
        + self.gammabar*self.sigma**2*(self.alpha*(1-self.gamma)-1)*self.E(l)**2*x**2)*l**(-1)*u*vF \
        + self.sigma**2*(self.E(l)*x - 1)*(self.alpha*(1-self.gammabar)-1)*self.E(l)*x*u**2*vC2/l
        return np.array([Mc, Ml])

    def M_bnd(self, cbar, u, v, vF, vB, vC2):
        mu_u, sig_u = self.mu_u(cbar,1), self.sigma_u(cbar,1)
        mu_theta, sig_theta = self.mu_theta(1), self.sigma_theta(1)
        return self.scrap - cbar*u + self.rho*u*(cbar**(1-self.gammabar)*vF - vB)/(self.gammabar - 1)

    #this is the quadratic characterizing the restricted-action allocation
    def Q(self,l):
        return self.E(l)**2*(self.gammabar-1)*(self.gammabar-1/2)*self.sigma**2*self.x(l)**2 + self.rho*(self.x(l) - 1)

    #following are expressions for taxation in competitive equilibrium.
    #this was the expression adopted in WP and WP-R. However, it is fragile and
    #therefore ultimately relegated to the appendix of subsequent versions.
    def tax_competitive(self,sigma,l):
        with np.errstate(divide='ignore',invalid='ignore'):
            A = self.rho**(-1)*self.E(l)**2*(self.gammabar-1)*(self.gammabar-1/2)*sigma**2
            x = (-1 + np.sqrt(1 + 4*A))/(2*A)
        x = np.nan_to_num(x,nan=1)
        num = self.gammabar*sigma**2*self.E(l)*x
        denom = self.rho - self.mu_theta(l) + self.gammabar*sigma**2*self.E(l)*x
        tau_k = num/denom
        tau_a = self.gammabar**2*sigma**2*self.E(l)**2*x**2 - self.rho*tau_k
        return tau_k, tau_a

    #exprssions for revenue raised as a fraction of wealth
    def revenue(self,sigma,l):
        with np.errstate(divide='ignore',invalid='ignore'):
            A = self.rho**(-1)*self.E(l)**2*(self.gammabar-1)*(self.gammabar-1/2)*sigma**2
            x = (-1 + np.sqrt(1 + 4*A))/(2*A)
        x = np.nan_to_num(x,nan=1)
        return self.gammabar*(self.gammabar-1)*sigma**2*self.E(l)**2*x**2

    def stat_dist(self,cbar,l,inj):
        b = 0*self.ugrid
        b[inj] = self.rhoD*self.Dt
        delta_m = self.rhoD - self.mu_theta(l)
        mu_m = (self.mu_u(cbar,l) + self.sigma_theta(l)*self.sigma_u(cbar,l))*self.ugrid
        sig_m = self.sigma_u(cbar,l)*self.ugrid
        pup = (self.Dt/self.Deltau**2)*(sig_m**2/2 + self.Deltau*np.maximum(mu_m,0))
        pdown = (self.Dt/self.Deltau**2)*(sig_m**2/2 + self.Deltau*np.maximum(-mu_m,0))
        pup[-1], pdown[0] = 0, 0
        pstay = 1 - pup - pdown
        pup, pdown, pstay = (1-delta_m*self.Dt)*pup, (1-delta_m*self.Dt)*pdown, (1-delta_m*self.Dt)*pstay
        P = self.T_func(np.arange(self.Nu-1), np.arange(self.Nu-1), pstay)
        row, column = np.arange(self.Nu-2), np.arange(1,self.Nu-1)
        P = P + self.T_func(row,column,pup[:-1])
        row, column = np.arange(1,self.Nu-1), np.arange(self.Nu-2)
        P = P + self.T_func(row,column,pdown[1:])
        return sp.linalg.spsolve(sp.eye(self.Nu-1) - P.T, b).reshape((self.Nu-1,))

    def rest_stat_y(self,beta,eta_E):
        RC_excess = 0*self.ugrid
        M, C = 0*self.ugrid, 0*self.ugrid
        for i in range(self.Nu-1):
            lr = self.rest_l[i]
            cr = self.rest(lr,self.ugrid[i])[0]
            mu_theta = self.mu_theta(lr)
            mu_c = (1-self.gammabar)*self.sigma**2*self.E(lr)**2*self.x(lr)**2/2
            M[i], C[i] = self.rhoD*(lr<1)/(self.rhoD - mu_theta), self.rhoD*cr/(self.rhoD - mu_c)
            RC_excess[i] = M[i]/(1-beta) - C[i] - self.ugrid[i]*(1-eta_E)/eta_E
        index = np.where(RC_excess<0)[0][0]
        y = self.ugrid[index]
        u = (1-beta)*(eta_E*M[index])**(-beta)
        return y, u, RC_excess, (M, C)

    def stat_y(self,beta,eta_E,cbar,l):
        RC_excess = 0*self.ugrid
        M, C = 0*self.ugrid, 0*self.ugrid
        for i in range(self.Nu-1):
            m = self.stat_dist(cbar,l,i)
            M[i], C[i] = np.sum(m*(l<1)), np.sum(cbar*self.ugrid*m)
            RC_excess[i] = M[i]/(1-beta) - C[i] - self.ugrid[i]*(1-eta_E)/eta_E
        index = np.where(RC_excess<0)[0][0]
        y = self.ugrid[index]
        u = (1-beta)*(eta_E*M[index])**(-beta)
        return y, u, RC_excess, (M, C)

    def tail(self,sigma,l):
        with np.errstate(divide='ignore',invalid='ignore'):
            A = self.rho**(-1)*self.E(l)**2*(self.gammabar-1)*(self.gammabar-1/2)*sigma**2
            x = (-1 + np.sqrt(1 + 4*A))/(2*A)
        x = np.nan_to_num(x,nan=1)
        z = 4*self.rhoD*(1 - 1/self.gammabar)*(2-1/self.gammabar)/(self.rho*(1-x))
        c = -self.gammabar/2 - (self.gammabar/2)*np.sqrt(1+z)
        theta = self.mu_theta(l)/sigma**2 - 1/2 - np.sqrt((self.mu_theta(l)/sigma**2 - 1/2)**2 + 2*self.rhoD/sigma**2)
        return c, theta

    def bigC(self,l):
        c = self.rest(l,1)[0]
        mu_c = (1-self.gammabar)*self.sigma**2*self.E(l)**2*self.x(l)**2/2
        return self.rhoD*c/(self.rhoD - mu_c)

    def T_func(self,A,B,C):
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.Nu-1,self.Nu-1))

    def kappa(self,eta_E,l):
        w = self.rhoD/(self.rhoD - (1-self.gammabar)*self.sigma**2*self.E(l)**2*self.x(l)**2/2)
        return eta_E*self.x(l)**(1/(1-self.gammabar))*l**(-self.alpha/(1-self.alpha)) + 1 - eta_E

    def Y(self,eta_E,l,Z,L,beta):
        return Z*(self.rhoD*eta_E/(self.rhoD - self.mu_theta(l)))**(1-beta)*L**beta

    def T_Y(self,eta_E,l,beta):
        return (1/self.kappa(eta_E,l) - beta)/self.rho

    def tau_d(self,eta_E,l,beta):
        return 1 - (1-beta)**(-1)*(self.rhoD*(self.rho - self.mu_theta(l))/(self.rho*(self.rhoD - self.mu_theta(l))))*(w - beta)*eta_E

    def D_Y(self,eta_E,l,beta):
        j = (self.x(l)**(self.gammabar/(1-self.gammabar))*l**(-self.alpha/(1-self.alpha))-self.kappa(eta_E,l))/self.kappa(eta_E,l)
        flow =  1 -beta - (self.rhoD/self.rho)*(((self.rho - self.mu_theta(l))/(self.rhoD - self.mu_theta(l)))*(j + 1-beta)*eta_E + (1-eta_E)*(1/self.kappa(eta_E,l) - beta))
        return flow/(self.rho-self.rhoD)

    def max_U(self,eta_E,beta):
        lgrid = np.linspace(self.llow, 1, 500)
        W = -(self.rhoD*eta_E/(self.rhoD - self.mu_theta(lgrid)))**(1-beta)/self.kappa(eta_E,lgrid)
        return lgrid[W.argmin()]

    #following checks concavity of value function as a function of UTILITY, not
    #as a function of NORMALIZED UTILITY. i.e. it checks concavity of V not v
    def check_concave(self,v):
        vp = (v[2:] - v[:-2])/(2*self.Deltau)
        vpp = (v[2:] - 2*v[1:-1] + v[:-2])/self.Deltau**2
        u = self.ugrid[1:-1]
        return self.gammabar*u*vp + u**2*vpp

    def K(self,l):
        return 1 - l + l*(l**(self.alpha*(self.gamma-1)/self.gammabar)-1)/(self.alpha*(self.gamma-1)/self.gammabar)

    def GLOBAL_check(self,l):
        return (self.alpha/(1-self.alpha))*(self.gammabar-1)*(self.gammabar-1/2)*(1-l)**4/(4*self.gammabar**2*self.K(l)**2) \
        + l*(1-l)**2/(2*self.gammabar*self.K(l)) - self.rho*self.alpha*self.sigma**2/((1-self.alpha)*(self.mu_0-self.mu_1)**2)

"""
To avoid confusion I treat the logarithmic case separately.
"""

class LOG(object):
    def __init__(self, alpha=0.33, rhoS=0.05, rhoD=0.05, sigma=0.3,
    mu_0 = 0.08, mu_1 = 0., umax = 6, Nu = 200, llow = 1/3, scrap = 0,
    maxiter = 20, tol = 10**-4, Dt = 10**(-6), nu = 0.25):
        self.alpha, self.nu = alpha, nu
        self.rhoS, self.rhoD, self.rho = rhoS, rhoD, rhoS + rhoD
        self.mu_1, self.mu_0, self.sigma = mu_1, mu_0, sigma
        self.umax, self.llow, self.lhigh = umax, llow, 1
        self.Deltau, self.Nu, self.Dt = (umax - 0)/Nu, Nu, Dt
        self.scrap, self.maxiter, self.tol = scrap, maxiter, tol
        self.ugrid = np.linspace(self.Deltau, self.umax - self.Deltau, self.Nu-1)
        self.rest_l = np.asarray([self.rest_pol(self.ugrid[i]) for i in range(self.Nu-1)])
        self.v_star = self.rest_PA(self.rest_l,self.ugrid)
        self.global_suff = self.rho < (1/self.alpha-1)*(self.mu_0 - self.mu_1)*self.llow
        self.finite = self.rhoD > self.mu_theta(self.llow)
        self.c_bnd = 0.5, 1.5*(self.rest(self.llow,1)[0])
        self.x0 = ((self.c_bnd[0] + self.c_bnd[1])/2, (self.llow + 1.0)/2)

    def E(self,l):
        return (l<1)*self.rho*self.alpha/((1-self.alpha)*(self.mu_0 - self.mu_1)*l)

    def mu_theta(self,l):
        return (self.mu_0 - (self.mu_0-self.mu_1)*l)

    def sigma_theta(self,l):
        return self.sigma*(l<1)

    def mu_u(self,cbar,l):
        return -self.rho*np.log(cbar) - (self.rho*self.alpha/(1-self.alpha))*np.log(l) \
        - self.mu_theta(l) + (self.sigma*self.E(l)-self.sigma_theta(l))**2/2 + self.sigma_theta(l)**2/2

    def sigma_u(self,cbar,l):
        return self.sigma*(self.E(l) - 1)*(l<1)

    def risk_adj(self,cbar,l):
        return self.mu_u(cbar,l) - self.sigma_u(cbar,l)**2/2 - (self.mu_theta(l) - self.sigma_theta(l)**2/2)

    def rest(self,l,u):
        ind = l<1
        slope = self.rho**(-1)*np.exp(self.sigma**2*self.E(l)**2/(2*self.rho))*l**(-self.alpha/(1-self.alpha))
        v_rest = (ind + (1-ind)*self.scrap)*(self.rho - self.mu_theta(l))**(-1) - slope*u
        c_rest = self.rho*slope*u
        return c_rest, v_rest, slope

    def rest_PA(self,l,u):
        ind = l<1
        return (ind + (1-ind)*self.scrap)*(self.rho - self.mu_theta(l))**(-1) - self.rest(l,1)[2]*u

    def rest_pol(self,u):
        Y = -self.rest_PA(np.linspace(self.llow, 1, 500),u)
        return np.linspace(self.llow, 1, 500)[Y.argmin()]

    def tran_func(self,cbar,l):
        tran_func = {}
        mu_u, sig_u = self.mu_u(cbar,l), self.sigma_u(cbar,l)
        mu_theta, sig_theta = self.mu_theta(l), self.sigma_theta(l)
        mu_hat1 = -self.rho*np.log(self.nu)-(self.rho*self.alpha/(1-self.alpha))*np.log(l) \
        + self.sigma**2*self.E(l)**2/2
        mu_hat2 = self.rho*np.log(cbar/self.nu) + self.mu_theta(l)
        u, du = self.ugrid, self.Deltau
        tran_func[(1)] = mu_hat1*u/du + sig_u**2*u**2/(2*du**2)
        tran_func[(0)] = -self.rho + mu_theta - (mu_hat1 + mu_hat2)*u/du - sig_u**2*u**2/du**2
        tran_func[(-1)] = mu_hat2*u/du + sig_u**2*u**2/(2*du**2)
        return tran_func

    def T_tran(self,cbar,l):
        #diagonal:
        row = np.arange(self.Nu-1)
        diag = self.tran_func(cbar,l)[(0)]
        T = self.T_func(row,row,diag)
        #up:
        row = np.arange(self.Nu-2)
        diag = self.tran_func(cbar,l)[(1)][:-1]
        T = T + self.T_func(row,row+1,diag)
        #down:
        row = np.arange(1,self.Nu-1)
        diag = self.tran_func(cbar,l)[(-1)][1:]
        T = T + self.T_func(row,row-1,diag)
        return T

    def v(self,cbar,l):
        ind = l<1
        b = (ind + (1-ind)*self.scrap) - cbar*self.ugrid
        b[-1] = b[-1] + self.tran_func(cbar,l)[(1)][-1]*self.rest_PA(1,self.umax)
        b[0] = b[0] + self.tran_func(cbar,l)[(-1)][0]*self.rest_PA(self.llow,0)
        return sp.linalg.spsolve(-self.T_tran(cbar,l), b)

    def polupdate(self,v):
        cbar, l, cl1 = np.zeros(self.Nu-1,), np.zeros(self.Nu-1,), np.zeros(self.Nu-1,)
        eval, eval_bnd = np.zeros(self.Nu-1,), np.zeros(self.Nu-1,)
        u, vbig = self.ugrid, np.zeros(self.Nu+1,)
        vbig[1:-1] = v
        vbig[0], vbig[-1] = self.rest_PA(self.llow,0), self.rest_PA(1,self.umax)
        vF, vB = (vbig[2:] - vbig[1:-1])/self.Deltau, (vbig[1:-1] - vbig[0:-2])/self.Deltau
        vC2 = (vbig[2:] - 2*vbig[1:-1] + vbig[0:-2])/self.Deltau**2
        x0 = (self.llow + 1.0)/2
        for i in range(self.Nu-1):
            f = lambda l: -self.G(l, u[i], v[i], vF[i], vB[i], vC2[i])
            f_jac = lambda l: -self.G_prime(l, u[i], v[i], vF[i], vB[i], vC2[i])
            res = minimize(f, x0, jac=f_jac, bounds=[(self.llow, 1.0)])
            l[i] = res.x[0]
            cbar[i] = np.minimum(np.maximum(-self.rho*vB[i], self.c_bnd[0]), self.c_bnd[1])
            eval[i] = self.G(l[i], u[i], v[i], vF[i], vB[i], vC2[i])
            ind_bnd = eval[i] < 0
            l[i] = l[i]*(1-ind_bnd) + ind_bnd
            x0 = l[i]
        l[-1], cbar[-1] = 1, 1
        return cbar, l

    def solve_PFI(self):
        v, i, eps = self.v_star, 1, 1
        while i < self.maxiter and eps > self.tol:
            tic = time.time()
            cbar, l = self.polupdate(v)
            toc = time.time()
            print("Time taken:", toc-tic)
            v1 = self.v(cbar,l)
            eps = np.sum(np.abs(v1-v))/self.Nu
            v, i = v1, i+1
            print("Difference in PFI:", eps, "Iterations:", i-1)
        return v

    def HJB_error(self,cbar,l,v):
        ind = l<1
        b = (ind + (1-ind)*self.scrap) - cbar*self.ugrid
        b[-1] = b[-1] + self.tran_func(cbar,l)[(1)][-1]*self.rest_PA(1,self.umax)
        b[0] = b[0] + self.tran_func(cbar,l)[(-1)][0]*self.rest_PA(self.llow,0)
        return b + self.T_tran(cbar,l)*v

    def G(self, l, u, v, vF, vB, vC2):
        ind = l<1
        return ind + (self.mu_0 - (self.mu_0 - self.mu_1)*l)*(v - u*vB) \
        + (- (self.rho*self.alpha/(1-self.alpha))*np.log(l) + self.sigma**2*self.E(l)**2/2)*u*vF \
        + self.sigma**2*(self.E(l) - ind)**2*u**2*vC2/2

    def G_prime(self, l, u, v, vF, vB, vC2):
        return - (self.mu_0 - self.mu_1)*(v - u*vB) \
        + (-self.rho*self.alpha/(1-self.alpha) - self.sigma**2*self.E(l)**2)*l**(-1)*u*vF \
        - self.sigma**2*(self.E(l) - 1)*self.E(l)*l**(-1)*u**2*vC2

    #following are expressions for taxation in competitive equilibrium.
    #this was the expression adopted in WP and WP-R. It is fragile and therefore
    #was ultimately relegated to the appendix.
    def tax_competitive(self,sigma,l):
        num = sigma**2*self.E(l)
        denom = self.rho - self.mu_theta(l) + sigma**2*self.E(l)
        tau_k = num/denom
        tau_a = sigma**2*self.E(l)**2 - self.rho*tau_k
        return tau_k, tau_a

    def stat_dist(self,cbar,l,inj):
        b = 0*self.ugrid
        b[inj] = self.rhoD*self.Dt
        delta_m = self.rhoD - self.mu_theta(l)
        mu_m = (self.mu_u(cbar,l) + self.sigma_theta(l)*self.sigma_u(cbar,l))*self.ugrid
        sig_m = self.sigma_u(cbar,l)*self.ugrid
        pup = (self.Dt/self.Deltau**2)*(sig_m**2/2 + self.Deltau*np.maximum(mu_m,0))
        pdown = (self.Dt/self.Deltau**2)*(sig_m**2/2 + self.Deltau*np.maximum(-mu_m,0))
        pup[-1], pdown[0] = 0, 0
        pstay = 1 - pup - pdown
        pup, pdown, pstay = (1-delta_m*self.Dt)*pup, (1-delta_m*self.Dt)*pdown, (1-delta_m*self.Dt)*pstay
        P = self.T_func(np.arange(self.Nu-1), np.arange(self.Nu-1), pstay)
        row, column = np.arange(self.Nu-2), np.arange(1,self.Nu-1)
        P = P + self.T_func(row,column,pup[:-1])
        row, column = np.arange(1,self.Nu-1), np.arange(self.Nu-2)
        P = P + self.T_func(row,column,pdown[1:])
        return sp.linalg.spsolve(sp.eye(self.Nu-1) - P.T, b).reshape((self.Nu-1,))

    def rest_stat_y(self,beta,eta_E):
        RC_excess = 0*self.ugrid
        M, C = 0*self.ugrid, 0*self.ugrid
        for i in range(self.Nu-1):
            lr = self.rest_l[i]
            cr = self.rest(lr,self.ugrid[i])[0]
            mu_theta = self.mu_theta(lr)
            M[i], C[i] = self.rhoD*(lr<1)/(self.rhoD - mu_theta), cr
            RC_excess[i] = M[i]/(1-beta) - C[i] - self.ugrid[i]*(1-eta_E)/eta_E
        index = np.where(RC_excess<0)[0][0]
        y = self.ugrid[index]
        u = (1-beta)*(eta_E*M[index])**(-beta)
        return y, u, RC_excess

    def stat_y(self,beta,eta_E,cbar,l):
        RC_excess = 0*self.ugrid
        M, C = 0*self.ugrid, 0*self.ugrid
        for i in range(self.Nu-1):
            m = self.stat_dist(cbar,l,i)
            M[i], C[i] = np.sum(m*(l<1)), np.sum(cbar*self.ugrid*m)
            RC_excess[i] = M[i]/(1-beta) - C[i] - self.ugrid[i]*(1-eta_E)/eta_E
        index = np.where(RC_excess<0)[0][0]
        y = self.ugrid[index]
        u = (1-beta)*(eta_E*M[index])**(-beta)
        return y, u, RC_excess

    #the expression c = - 1/2 - (1/2)*np.sqrt(1+z) with 2*self.rhoD/(sigma**2*self.E(l)**2) is incorrect. it is missing a factor of two.
    def tail(self,sigma,l):
        z = 2*self.rhoD/(sigma**2*self.E(l)**2)
        c = - 1/2 - np.sqrt(1/4+z)
        theta = self.mu_theta(l)/sigma**2 - 1/2 - np.sqrt((self.mu_theta(l)/sigma**2 - 1/2)**2 + 2*self.rhoD/sigma**2)
        return c, theta

    def bigC(self,l):
        return self.rest(l,1)[0]

    def T_func(self,A,B,C):
        return sp.coo_matrix((C.reshape(np.size(C),),(A.reshape(np.size(A),),B.reshape(np.size(B),))),shape=(self.Nu-1,self.Nu-1))

    def kappa(self,eta_E,l):
        return eta_E*self.rest(l,1)[0] + 1 - eta_E

    def Y(self,eta_E,l,Z,L,beta):
        return Z*(self.rhoD*eta_E/(self.rhoD - self.mu_theta(l)))**(1-beta)*L**beta

    def T_Y(self,eta_E,l,beta):
        return (1/self.kappa(eta_E,l) - beta)/self.rho

    def tau_d(self,eta_E,l,beta):
        return 1 - (1-beta)**(-1)*(self.rhoD*(self.rho - self.mu_theta(l))/(self.rho*(self.rhoD - self.mu_theta(l))))*(w - beta)*eta_E

    def D_Y(self,eta_E,l,beta):
        j = (self.rest(l,1)[0]-self.kappa(eta_E,l))/self.kappa(eta_E,l)
        flow = 1 - beta - (self.rhoD/self.rho)*(((self.rho - self.mu_theta(l))/(self.rhoD - self.mu_theta(l)))*(j + 1-beta)*eta_E + (1-eta_E)*(1/self.kappa(eta_E,l) - beta))
        return flow/(self.rho-self.rhoD)

    def max_U(self,eta_E,beta):
        lgrid = np.linspace(self.llow, 1, 500)
        W = -(self.rhoD*eta_E/(self.rhoD - self.mu_theta(lgrid)))**(1-beta)/self.kappa(eta_E,lgrid)
        return lgrid[W.argmin()]
