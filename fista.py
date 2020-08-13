from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt, log, pi, exp, log10
import math
from math import e
import time

# assume I = Phi + kappa*|.|_1

def shrinkage(z, cutoff):
    retvals = np.zeros((len(z),))
    #cutoff = 2*tau*gamma**2*sqrt(kappa)*5000
    #cutoff = 2*steptau*gamma**2*kappa
    for k in range(len(z)):
        if z[k] >= cutoff:
	        retvals[k] = z[k]-cutoff
        elif z[k] >= -cutoff:
	        retvals[k] = 0
        else: 
	        retvals[k] = z[k]+cutoff
    return retvals



def FISTA(x0, Phi_fnc, DPhi_fnc, cutoffmultiplier, alpha0=1.0, eta=0.5, N_iter=500, c=1.0, showDetails=False, DPhi_has_both=False, save_xk=False, noNormLastIndices=0):
    # assumes that the correct norm on the parameter is the plain l1 norm. Scale parameters to achieve this.
    # minimizes ||Phi(u)||_2^2 + cutoffmultiplier * ||u||_1
    
    
    # if DPhi_has_both is True, DPhi_fnc is assumed to consist of Phi and DPhi (as a tuple)
    
    # noNormLastIndices specifies how many indices (counted from the last entry) should not be reduced by the prox operator (because they are not to be inserted into the norm)
    
    start = time.time()
    if save_xk:
        xk = np.zeros((N_iter, x0.size))
        xk[0, :] = x0
    xk_now = x0
    
    yk = x0
    tk = np.zeros((N_iter,))
    tk[1] = 1
    Is = np.zeros((N_iter,))
    Phis = np.zeros((N_iter,))
    if DPhi_has_both:        
        Phis[0], DPhi = DPhi_fnc(x0)
    else:
        Phis[0] = Phi_fnc(x0)
        DPhi = DPhi_fnc(x0)
    Is[0] = Phis[0] + cutoffmultiplier*np.sum(np.abs(x0))
    alpha = alpha0
	
    # for running diagnostics:
    num_backtrackings = np.zeros((N_iter,))
    alpha = alpha0*eta
    for k in range(1, N_iter):
        if showDetails:
            print("Iteration " + str(k) + ": " + "{:.2f}".format(time.time()-start) + "s")
        
        
        alpha = alpha/eta # go back one backtracking step each iteration
        #proposal = shrinkage(yk[k, :] - alpha*DPhi, cutoffmultiplier*alpha)
        if noNormLastIndices > 0:
            PhiDescentStep = yk - alpha*DPhi
            part1 = shrinkage(PhiDescentStep[0:-noNormLastIndices], cutoffmultiplier*alpha)
            part2 = PhiDescentStep[-noNormLastIndices:]
            proposal = np.concatenate((part1, part2))
        else:
            proposal = shrinkage(yk - alpha*DPhi, cutoffmultiplier*alpha)
        max_backtrack = 20;
        Phiprop = Phi_fnc(proposal)
        #Phiyk = Phi_fnc(yk[k,:])
        Phiyk = Phi_fnc(yk)
        #while np.isnan(Phiprop) or Phiprop + cutoffmultiplier*np.sum(np.abs(proposal)) > Phiyk + np.dot(DPhi.T, proposal-yk[k, :]) + 1.0/(2.0*alpha*c)*np.dot((proposal-yk[k, :]).T, proposal-yk[k, :]) + cutoffmultiplier*np.sum(np.abs(yk[k,:])):
        while np.isnan(Phiprop) or Phiprop + cutoffmultiplier*np.sum(np.abs(proposal)) > Phiyk + np.dot(DPhi.T, proposal-yk) + 1.0/(2.0*alpha*c)*np.dot((proposal-yk).T, proposal-yk) + cutoffmultiplier*np.sum(np.abs(yk)):  
            alpha = alpha*eta
            #proposal = shrinkage(yk[k, :] - alpha*DPhi,  cutoffmultiplier*alpha)
            if noNormLastIndices > 0:
                PhiDescentStep = yk - alpha*DPhi
                part1 = shrinkage(PhiDescentStep[0:-noNormLastIndices], cutoffmultiplier*alpha)
                part2 = PhiDescentStep[-noNormLastIndices:]
                proposal = np.concatenate((part1, part2))
            else:
                proposal = shrinkage(yk - alpha*DPhi, cutoffmultiplier*alpha)
            Phiprop = Phi_fnc(proposal)
            max_backtrack -= 1
            num_backtrackings[k] += 1
            if max_backtrack <= 0:
                # restart
                print("restart backtracking")
                alpha = alpha0*eta
                proposal = xk_now
                yk = xk_now
                break
        
        if save_xk:
            xk_new = proposal
            xk[k, :] = xk_new
        else:
            xk_new = proposal
        
        Phis[k] = Phi_fnc(xk_new)
        Is[k] = Phis[k] + cutoffmultiplier*np.sum(np.abs(xk_new))
        if showDetails:
        	print("I = " + str(Is[k]) +", num_bt = " + str(num_backtrackings[k]))
        if k < N_iter-1: # preparation for next step only needed up to penultimate iteration
            tk[k+1] = 0.5 * (1.0 + sqrt(1.0 + 4.0*tk[k]**2))
            yk_new = xk_new + (tk[k]-1)/tk[k+1] * (xk_new - xk_now)
            xk_now = xk_new
            yk = yk_new
            if DPhi_has_both:
                _, DPhi = DPhi_fnc(yk)
            else:
                DPhi = DPhi_fnc(yk)
    if save_xk:
        result = {"xOpt": xk_new, "xk": xk, "Is": Is, "Phis": Phis, "num_backtrackings": num_backtrackings}
    else:
        result = {"xOpt": xk_new, "Is": Is, "Phis": Phis, "num_backtrackings": num_backtrackings}
    end = time.time()
    if showDetails:
        print("Took " + str(end-start) + " seconds")
        print("Reduction of function value from " + str(Is[0]) + " to " + str(Is[-1]))
        print("Function value consists of")
        print("Phi(u)  = " + str(Phis[-1]))
        print("norm(u) = " + str(Is[-1] - Phis[-1]))
    return result

def FISTA_greedy(x0, Phi_fnc, DPhi_fnc, cutoffmultiplier, gamma0=1.0, N_iter=500, S=3):
    c = 1.0
    gamma = gamma0
    xk = x0
    xkm1 = x0
    Is = np.zeros((N_iter,))
    Phis = np.zeros((N_iter,))
    Phis[0] = Phi_fnc(x0)
    Is[0] = Phis[0] + cutoffmultiplier*np.sum(np.abs(x0))
    for k in range(0, N_iter-1):
        yk = xk + (xk - xkm1)
        Phiyk = Phi_fnc(yk)
        DPhi = DPhi_fnc(yk)
        xkp1 = shrinkage(yk - gamma*DPhi, cutoffmultiplier*gamma)
        if k == 0:
            x1 = xkp1
        
    
        if np.dot(yk-xkp1, xkp1-xk) >= 0:
            print("restart in k = " + str(k))
            yk = xk
        
        if (np.dot(xkp1-xk, xkp1-xk) >= S**2 * np.dot(x1-x0, x1-x0)):
            print("safeguard case! k = " + str(k))
            gamma = 0.7*gamma
            xkp1 = shrinkage(yk - gamma*DPhi, cutoffmultiplier*gamma)
            while (np.dot(xkp1-xk, xkp1-xk) >= S**2 * np.dot(x1-x0, x1-x0)):
                gamma = 0.7*gamma
                xkp1 = shrinkage(yk - gamma*DPhi, cutoffmultiplier*gamma)
           
        Phikp1 = Phi_fnc(xkp1)
        Phis[k+1] = Phikp1
        Is[k+1] = Phis[k+1] + cutoffmultiplier*np.sum(np.abs(xkp1))
        xkm1 = xk
        xk = xkp1
        
    result = {"xOpt": xk, "Is": Is, "Phis": Phis}  
    return result  

def FISTA_greedy_BT(x0, Phi_fnc, DPhi_fnc, cutoffmultiplier, gamma0=1.0, eta=0.7, N_iter=500):
    c = 1.0
    gamma = gamma0
    eta = 0.5
    xk = x0
    xkm1 = x0
    Is = np.zeros((N_iter,))
    Phis = np.zeros((N_iter,))
    Phis[0] = Phi_fnc(x0)
    Is[0] = Phis[0] + cutoffmultiplier*np.sum(np.abs(x0))
    for k in range(0, N_iter-1):
        yk = xk + (xk - xkm1)
        Phiyk = Phi_fnc(yk)
        DPhi = DPhi_fnc(yk)
        proposal = shrinkage(yk - gamma*DPhi, cutoffmultiplier*gamma)
        Phiprop = Phi_fnc(proposal)
        
        
        max_backtrack = 10;
        while np.isnan(Phiprop) or Phiprop + cutoffmultiplier*np.sum(np.abs(proposal)) > Phiyk + np.dot(DPhi.T, proposal-yk) + 1.0/(2.0*gamma*c)*np.dot((proposal-yk).T, proposal-yk) + cutoffmultiplier*np.sum(np.abs(yk)):  
            print("bt")
            gamma = gamma*eta
            proposal = shrinkage(yk - gamma*DPhi, cutoffmultiplier*gamma)
            Phiprop = Phi_fnc(proposal)
            max_backtrack -= 1
            if max_backtrack <= 0:
                # restart
                print("restart backtracking")
                gamma = gamma0*eta
                proposal = xk
                yk = xk
                break
        xkp1 = proposal
    
        if np.dot(yk-xkp1, xkp1-xk) >= 0:
            print("restart")
            yk = xk
           
        Phis[k+1] = Phiprop
        Is[k+1] = Phis[k+1] + cutoffmultiplier*np.sum(np.abs(xkp1))
        xkm1 = xk
        xk = xkp1
        
    result = {"xOpt": xk, "Is": Is, "Phis": Phis}  
    return result    
	
if __name__ == "__main__":
    N = 100
    np.random.seed(1992)
    xs_obs = np.random.uniform(0, 3, (N,))

    N_modes = 19
    kappa = 20.0
    gamma = 0.5


    def f(u, xs):
	    temp = u[0]
	    for k in range(N_modes):
		    temp += u[k+1] * 1/(k+1)*np.cos((k+1)*xs)
	    for k in range(N_modes):
		    temp += u[k+N_modes+1] * 1/(k+1)* np.sin((k+1)*xs)
	    return temp

    A = np.zeros((N, 2*N_modes+1))
    A[:, 0] = np.ones((N,))
    for k in range(N_modes):
	    A[:, k+1] =  1/(k+1)*np.cos((k+1)*xs_obs)
    for k in range(N_modes):
	    A[:, k+N_modes+1] =  1/(k+1)*np.sin((k+1)*xs_obs)

    def norm1(u):
	    return kappa*np.sum(np.abs(u))
    def Misfit(u, y):
	    misfit = y - np.dot(A, u)
	    return 1.0/(2.0*gamma**2)*np.dot(misfit.T, misfit) 
    def FncL1(u, y):
	    return Misfit(u, y) + norm1(u)

    DMisfit = lambda u, y: gamma**(-2)*np.dot(A.T, np.dot(A, u)-y)

    uTruth = np.array([-1.0, 1.0, -1.3, -0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 2.2, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    x = np.arange(0, 3, 0.01)
    y = f(uTruth, x)

    obs = f(uTruth, xs_obs) + np.random.normal(0, gamma, (len(xs_obs),))

    plt.figure(1);plt.ion()
    plt.plot(x, y, 'g', label="true")
    plt.plot(xs_obs, obs, 'rx', label="data")
    plt.show()
    
    result = FISTA(np.zeros((uTruth.size,)), lambda x: Misfit(x, obs), lambda x: DMisfit(x, obs), 2*gamma**2*kappa, alpha0=0.003, eta=0.5, N_iter=100, showDetails=True)
    result_greedy = FISTA_greedy(np.zeros((uTruth.size,)), lambda x: Misfit(x, obs), lambda x: DMisfit(x, obs), 2*gamma**2*kappa, gamma0=0.002, N_iter=100)
    
    x0 = np.zeros((uTruth.size,))
    Phi_fnc = lambda x: Misfit(x, obs)
    DPhi_fnc = lambda x: DMisfit(x, obs)
    cutoffmultiplier = 2*gamma**2*kappa
    alpha = 0.00001
    N_iter = 500

    xOpt = result["xOpt"]
    xOpt_greedy = result_greedy["xOpt"]
    Is = result["Is"]
    Is_greedy = result_greedy["Is"]
    Phis = result["Phis"]
    Phis_greedy = result_greedy["Phis"]
    plt.plot(x, f(xOpt, x), 'b', label="opt")
    plt.plot(x, f(xOpt_greedy, x), 'k', label="opt_greedy")
    plt.legend()

    plt.figure();
    plt.subplot(311)
    plt.semilogy(Is, 'b-', label="Is")
    plt.semilogy(Is_greedy, 'k-', label="Is_greedy")
    plt.legend()
    plt.subplot(312)
    plt.semilogy(Phis, 'b-', label="Phis")
    plt.semilogy(Phis_greedy, 'k-', label="Phis_greedy")
    plt.legend()
    plt.subplot(313)
    plt.semilogy(Is-Phis, 'b-', label="norms")
    plt.semilogy(Is_greedy-Phis_greedy, 'k-', label="norms_greedy")
    plt.legend()


    print("ground truth: I = " + str(FncL1(uTruth, obs)) + " = " + str(Misfit(uTruth, obs)) + " (Phi) + " + str(norm1(uTruth)) + " (norm)")
    print("optimizer: I = " + str(FncL1(xOpt, obs)) + " = " + str(Misfit(xOpt, obs)) + " (Phi) + " + str(norm1(xOpt)) + " (norm)")

    plt.figure();
    plt.plot(uTruth, '.-g')
    plt.plot(xOpt, '.-k')
    plt.plot(xOpt_greedy, '--k')
	

	
