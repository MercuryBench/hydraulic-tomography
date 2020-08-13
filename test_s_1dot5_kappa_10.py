import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from grid import RectGrid

from math import e, log10
import haarWavelet as hw
import scipy.io as spio

from fwd_multellipt import *
from matplotlib import ticker

from fista import *

import time

np.random.seed(199)

plt.ion()
plt.show()


s = 1.5 # prior regularity (change)
kappa = 10.0 # prior factor (change)


sigNoise = 1e-6 # noise standard deviation (redundant parameter, don't change)




def waveletAnalysisFunction(fnc):
    MM = int(math.sqrt(len(fnc)))
    return hw.waveletanalysis2d(np.reshape(fnc, (MM,MM)))

def waveletSynthesisFunction(wc):
    return np.reshape(hw.waveletsynthesis2d(wc), (-1,))


def scaleWavelet(wc_unp, s):
    wc = hw.packWavelet(wc_unp)
    num_J = len(wc)
    mult_vec = np.zeros((2**(2*(num_J-1)),))
    mult_vec[0] = 1.0
    for j in range(1, num_J):
        mult_vec[2**(2*j-2):2**(2*j)] = np.concatenate((2**(s*j)*np.ones((2**(2*j-2),)),2**(s*j)*np.ones((2**(2*j-2),)),2**(s*j)*np.ones((2**(2*j-2),)) ))
    return mult_vec*wc_unp

def unscaleWavelet(wc_unp_scaled, s):
    wc = hw.packWavelet(wc_unp_scaled)
    num_J = len(wc)
    mult_vec = np.zeros((2**(2*(num_J-1)),))
    mult_vec[0] = 1.0
    for j in range(1, num_J):
        mult_vec[2**(2*j-2):2**(2*j)] = np.concatenate((2**(-s*j)*np.ones((2**(2*j-2),)),2**(-s*j)*np.ones((2**(2*j-2),)),2**(-s*j)*np.ones((2**(2*j-2),)) ))
    
    return mult_vec*wc_unp_scaled
    
	
if __name__ == "__main__":
    from math import sin, pi, cos
    tol = 1e-8
    def ind_dir(vec):
	    if (vec[0] < tol or vec[0] > 1-tol or vec[1] > 1-tol) and not (vec[1] < tol):
	        return 1
	    else:  
	        return 0

    def ind_neum(vec):
	    if vec[1] < tol:
	        return 1
	    else:
	        return 0

    def g_dir(vec):
	    return vec[:,0]*0.0#-np.cos(2*pi*vec[:,1]) + vec[:,0]**2# + (1 + vec[:,0]**2 + 2 * vec[:,1] ** 2)


    g_neum = lambda vec: 0

    bv = BoundaryValues(ind_Dir=ind_dir, ind_Neum=ind_neum, g_Dir=g_dir, g_Neum=g_neum)

	
    def coeff_u_01(vec):
        bg = (np.sin(3*vec[0])+np.sin(2*vec[0]))*0.5/log10(e)
        if (vec[0]-1)**2 + vec[1]**2 <= 0.65**2 and (vec[0]-1)**2 + vec[1]**2 > 0.45**2 and vec[0] < 0.85:
            return bg -2/log10(e)
        elif vec[0]-0.2*vec[1] < 0.345 and vec[0]-0.2*vec[1] >= 0.222 and vec[1] >= 0.625:
            return bg - 1.5/log10(e)
        else:
            return bg

    def coeff_f(vec, pos, sigx, sigy, strength):
        #return -2.0 - 4*pi**2*np.cos(2*pi*vec[:,1])
        return strength*np.exp(-((vec[:,0]-pos[0])**2)/(2*sigx**2)-((vec[:,1]-pos[1])**2)/(2*sigy**2))

    N = 5
    MM = 2**N
    rg = RectGrid(0, 160.0, MM, 0, 76, MM, bv)
    stratography = rg.orderByGroup(np.flipud(spio.loadmat("stratography.mat", squeeze_me=True)["B"].T.flatten())).astype(int)
    umapping = np.random.normal(0, 1, (18,))
    uTruth = np.array([umapping[s] for s in stratography])
    kTruth = np.exp(uTruth)
    
	#indobs = range(3, MM*MM-4*MM, 5)
    """ind_2d = np.zeros((MM-2, MM-2))
    for kk in range(1, MM-2, 4):
        for ll in range(1, MM-2, 4):
            ind_2d[kk, ll] = 1

    ind_1d = np.reshape(ind_2d, (-1,))
    indobs = np.nonzero(ind_1d)[0]"""
    
    strength = 10*8.27e-6/0.102 # siehe Wolfgangs Code
    
    
    #source_pos = np.array([[0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8]])
    # 1.: actual (real sandbox) locations
    
    measXloc = np.array([27.5, 48, 68.4, 93.1, 114, 134])
    
    measYloc = np.flipud(np.array([8.33, 16.5, 24.8, 33, 41.3, 49.6, 57.8, 66.1]))
    Xlocmg, Ylocmg = np.meshgrid(measXloc, measYloc)
    XX = Xlocmg.flatten()
    YY = Ylocmg.flatten()
    well_pos_data = np.array([[xx, yy] for xx, yy in zip(XX, YY)])
    
    # 2.: closest indices to real sandbox locations
    well_pos_grid_ind = np.zeros((well_pos_data.shape[0],), dtype=np.int32)
    
    ipts, npts, dpts = rg.getPoints()
    for rowind in range(well_pos_data.shape[0]):
        spos = well_pos_data[rowind, :]
        well_pos_grid_ind[rowind] = int(np.argmin(np.sum((ipts-spos)**2, axis=1)))
    
    # 3.: measurement location compatible with grid
    well_pos_grid = ipts[well_pos_grid_ind]
    
    source_ind =  [1, 4, 13, 16, 31, 34, 43, 46]
    source_pos = well_pos_grid[source_ind]
    #indobs = well_pos_grid_ind[source_ind]
    
    # make individual observation by dropping the source position
    #indobslist = [np.concatenate((indobs[0:kk], indobs[kk+1:])) for kk in range(len(source_ind))]
    indobslist = []
    for nn, ind in enumerate(source_pos):
        indobslist.append(np.concatenate((well_pos_grid_ind[0:source_ind[nn]], well_pos_grid_ind[source_ind[nn]+1:])))
    
    # width of pump is one half of a cell width
    sigx = 0.5*(rg.x2-rg.x1)/(MM-1)
    sigy = 0.5*(rg.y2-rg.y1)/(MM-1)
    ep = []
    
	
    #ep = [MultEllipticalProblem(rg, kTruth, lambda vec, temp=ind: coeff_f(vec, source_pos[temp, :], sigx, sigy, strength), bv, indobslist[ind]) for ind in range(source_pos.shape[0])]
    ep = MultEllipticalProblem(rg, kTruth, [lambda vec, temp=ind: coeff_f(vec, source_pos[temp, :], sigx, sigy, strength) for ind in range(source_pos.shape[0])] , bv, [indobslist[ind] for ind in range(source_pos.shape[0])])  
    
   
    
    
    bdpts = np.array(ep.grid.getBoundaryPointsByFnc(ep.boundaryValues.ind_Dir))

    xibars = []
    obss = []

    plt.figure(); plt.ion()

    xibars = ep.fwdOp()
    noise = [np.random.normal(0, sigNoise, (len(indobslist[kk]),)) for kk in range(source_pos.shape[0])]
    obss = [xibars[kk][indobslist[kk]] + noise[kk] for kk in range(len(xibars))]
    for kk in range(len(xibars)):
        xibar = xibars[kk]
        obs = obss[kk]
        plt.subplot(4, 2, kk+1)
        ext = [rg.x1, rg.x2, rg.y1, rg.y2]
        pvals = np.reshape(rg.orderSpatially(np.concatenate((xibar, ep._xihat))), (rg.Nx, rg.Ny))
        plt.imshow(np.rot90(pvals), extent=ext, cmap=plt.cm.viridis, interpolation='none')
        ipts, npts, dpts = rg.getPoints()
        freepts = np.concatenate((ipts, npts), axis=0)
        vmin1 = np.min(obs)
        vmin2 = np.min(pvals)
        vmin = min(vmin1, vmin2)
        vmax1 = np.max(obs)
        vmax2 = np.max(pvals)
        vmax = max(vmax1, vmax2)
        v1 = ipts[indobslist[kk], 0]
        v2 = ipts[indobslist[kk], 1]
        plt.scatter(v1, v2, s=20, c=obs, vmin=vmin, vmax=vmax , cmap=plt.cm.viridis, edgecolors="black")
        cb = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()
        plt.axis("off")
        
        
        
    plt.savefig("simulations/p_groundtruth.pdf", bbox_inches = 'tight', pad_inches = 0)
    
    # plot logpermeability
    plt.figure()
    logkvals = np.reshape(rg.orderSpatially(np.log(kTruth)), (rg.Nx, rg.Ny))
    im = plt.imshow(np.rot90(logkvals), extent=ext, cmap=plt.cm.viridis, interpolation='none')
    plt.colorbar(im, fraction=0.04, pad=0.04, aspect=10)
    plt.savefig("simulations/u_groundtruth.pdf", bbox_inches = 'tight', pad_inches = 0)

    
    
    
    
    def scaledwcunpToFE(wc_unp):
        return rg.orderByGroup(np.reshape(hw.waveletsynthesis2d(hw.packWavelet(unscaleWavelet(wc_unp, s))), (-1,)))

    def FEToscaledwcunp(fnc):
        MM = int(math.sqrt(len(fnc)))
        return scaleWavelet(hw.unpackWavelet(hw.waveletanalysis2d(np.reshape(rg.orderSpatially(fnc), (MM,MM)))), s)
    
    def scaledFETowcunp(fnc):
        MM = int(math.sqrt(len(fnc)))
        return unscaleWavelet(hw.unpackWavelet(hw.waveletanalysis2d(np.reshape(rg.orderSpatially(fnc), (MM,MM)))), s)*MM*MM
    
    def wcunpToFE(wc_unp):
        return rg.orderByGroup(np.reshape(hw.waveletsynthesis2d(hw.packWavelet(wc_unp)), (-1,)))

    def FETowcunp(fnc):
        MM = int(math.sqrt(len(fnc)))
        return hw.unpackWavelet(hw.waveletanalysis2d(np.reshape(rg.orderSpatially(fnc), (MM,MM))))
    
    
    
    def q_onscaled(u_wc_unp, obss, returnSum=True):
        return ep.qMisfitLogPermeability(scaledwcunpToFE(u_wc_unp), obss, returnSum=returnSum)
    
    def q_dq_onscaled(u_wc_unp, obss, returnSum=True):
        q, dq = ep.q_dqMisfitLogPermeability(scaledwcunpToFE(u_wc_unp), obss, returnSum=returnSum)
        if returnSum:
            dq_scaled = scaledFETowcunp(dq)
        else:
            dq_scaled = [scaledFETowcunp(dq_i) for dq_i in dq]
        return q, dq_scaled
    
    
    alpha = 0.5;
    NOpt = 50
    
    u0 = np.zeros(kTruth.shape)
    
    wc = hw.packWavelet(FEToscaledwcunp(u0))
    num_J = len(wc)
    mult_vec = np.zeros((2**(2*(num_J-1)),))
    mult_vec[0] = 1.0
    for j in range(1, num_J):
        mult_vec[2**(2*j-2):2**(2*j)] = np.concatenate((2**(s*j)*np.ones((2**(2*j-2),)),2**(s*j)*np.ones((2**(2*j-2),)),2**(s*j)*np.ones((2**(2*j-2),)) ))
    import scipy.optimize as opt
    
    u0_wc_unp = FEToscaledwcunp(u0)
    uTruth = np.log(kTruth)
    uTruth_wc_unp = FEToscaledwcunp(uTruth)
    
    import time as time
    s1 = time.time()
    q_onscaled(uTruth_wc_unp, obss)
    s2 = time.time()
    q_dq_onscaled(uTruth_wc_unp, obss)
    s3 = time.time()
    print("fwd: " + str(s2-s1))
    print("d_fwd: " + str(s3-s2))
    
    I = lambda x: q_onscaled(x, obss) + 2*sigNoise**2*kappa*np.sum(np.abs(x))
    
    
    """ for approximation test of the discrete adjoint gradient. Uncomment to debug
    # test approximation (individually)
    fnc = lambda x: q_onscaled(x, obss, False)
    Dfnc = lambda x: q_dq_onscaled(x, obss, False)
    
    uApprox = u0_wc_unp

    f0s, Df0s = Dfnc(uApprox)
    Df0s = np.stack(Df0s)

    grad = np.sum(Df0s, axis=0)


    direc = 0.3*(uTruth_wc_unp - uApprox)
    rs = np.linspace(-1, 1, 21)
    fncs = np.zeros((8, 21))
    fncs_lin = np.zeros((8, 21))
    for n, r in enumerate(rs):
        fncs[:, n] = fnc(uApprox+r*direc)
        fncs_lin[:, n] = f0s + r*np.dot(Df0s, direc)

    plt.figure()
    for n in range(8):
        plt.subplot(4,2,n+1)
        plt.plot(rs, fncs[n, :])
        plt.plot(rs, fncs_lin[n, :])

    plt.figure()
    plt.plot(rs, np.sum(fncs-fncs_lin, axis=0))"""
    
    
    
    
    result = FISTA(u0_wc_unp, lambda x: q_onscaled(x, obss), lambda x: q_dq_onscaled(x, obss), 2*sigNoise**2*kappa, alpha0=1000, eta=0.7, N_iter=2000, showDetails=True, DPhi_has_both=True, save_xk=False)
    
    

    


    #xk = result["xk"]
    Is = result["Is"]
    Phis = result["Phis"]

    uOpt_wc_unp = result["xOpt"]
    
    
    print("note: I is a factor of 2*sigma**2 smaller than 'actual' I")


    plt.figure();
    plt.subplot(311)
    plt.semilogy(Is, 'k-', label="Is")
    plt.legend()
    plt.subplot(312)
    plt.semilogy(Phis, 'k-', label="Phis")
    plt.legend()
    plt.subplot(313)
    plt.plot(Is-Phis, 'k-', label="norms")
    plt.legend()
    plt.savefig("simulations/errornorms.pdf", bbox_inches = 'tight', pad_inches = 0)

    uOpt_spatial_vals = hw.waveletsynthesis2d(hw.packWavelet(unscaleWavelet(uOpt_wc_unp, s)))
    uOpt = scaledwcunpToFE(uOpt_wc_unp)
    #PhiUOpt, wnUOpt = I(uOpt_wc_unp, obs, True)
    #PhiUTruth, wnUTruth = I(uTruth_wc_unp, obs, True)
    #print("uOpt: Phi = " + str(PhiUOpt) + ", wavelet norm = " + str(wnUOpt))
    #print("uTruth: Phi = " + str(PhiUTruth) + ", wavelet norm = " + str(wnUTruth))      
    kOpt = np.exp(uOpt)
    plt.figure(); plt.ion()

    ep.set_coeff_k(kOpt, True)
    xibars = ep.fwdOp()

    for kk in range(len(xibars)):
        xibar = xibars[kk]
        obs = obss[kk]
        plt.subplot(4, 2, kk+1)
        ext = [rg.x1, rg.x2, rg.y1, rg.y2]
        pvals = np.reshape(rg.orderSpatially(np.concatenate((xibar, ep._xihat))), (rg.Nx, rg.Ny))
        plt.imshow(np.rot90(pvals), extent=ext, cmap=plt.cm.viridis, interpolation='none')
        ipts, npts, dpts = rg.getPoints()
        freepts = np.concatenate((ipts, npts), axis=0)
        vmin1 = np.min(obs)
        vmin2 = np.min(pvals)
        vmin = min(vmin1, vmin2)
        vmax1 = np.max(obs)
        vmax2 = np.max(pvals)
        vmax = max(vmax1, vmax2)
        v1 = ipts[indobslist[kk], 0]
        v2 = ipts[indobslist[kk], 1]
        plt.scatter(v1, v2, s=20, c=obs, vmin=vmin, vmax=vmax , cmap=plt.cm.viridis, edgecolors="black")
        cb = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()
        plt.axis("off")


    plt.savefig("simulations/p_MAP.pdf", bbox_inches = 'tight', pad_inches = 0)

    plt.figure()
    logkvals = np.reshape(rg.orderSpatially(np.log(kOpt)), (rg.Nx, rg.Ny))
    im = plt.imshow(np.rot90(logkvals), extent=ext, cmap=plt.cm.viridis, interpolation='none')
    plt.colorbar(im, fraction=0.04, pad=0.04, aspect=10)
    plt.savefig("simulations/u_MAP.pdf", bbox_inches = 'tight', pad_inches = 0)
    #for kk in range(source_pos.shape[0]):
    #    xibar = ep[kk].fwdOp(kOpt)
    #    ep[kk].plotSolAndPerm(kOpt,  np.concatenate((xibar, xihat)), obss[kk], dim3 = False)



    # save data
    import pickle

    with open("simulations/data.pickle", 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
