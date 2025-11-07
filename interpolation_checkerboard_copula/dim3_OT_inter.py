# -----------------------------------------------------------------------------
# File: dim3_OT_inter.py
#
# This Python implementation is translated and adapted from the original MATLAB
# code in the repository "2013-SIIMS-ot-splitting" by N. Papadakis, G. Peyré, E. Oudet.  
# Original repository: https://github.com/gpeyre/2013-SIIMS-ot-splitting  
#
# Reference:
#   Papadakis, N., Peyré, G., & Oudet, E. (2014).
#   "Optimal Transport with Proximal Splitting."
#   SIAM Journal on Imaging Sciences, 7(1), 212–238.
#
# Original copyright: (c) 2009 Gabriel Peyré 
#
# License / Use terms:
#   - This code follows the original copyright notice and respects the author's
#     rights as declared in the original repository.
#   - If the original repository is later confirmed to have a specific open license
#     (e.g. MIT, BSD, GPL), this code should be distributed under compatible terms.
#
# -----------------------------------------------------------------------------

import numpy as np
import cmath
import math
from scipy.fft import fft, ifft
import scipy
from scipy.stats import norm
from scipy.stats import multivariate_normal
import warnings

class staggered:
    def __init__(self, dimvect):
        self.dim = dimvect
        self.dims = []
        self.M = []
        
        for k in range(len(dimvect)):
            lt = dimvect.copy()
            lt[k] += 1
            self.M.append(np.zeros(lt))
            self.dims.append(lt)

def interp(U):
    V = np.concatenate(((U.M[0][:-1, :, :] + U.M[0][1:, :, :, :])[np.newaxis,:, :, :, :], 
                (U.M[1][:, :-1, :, :] + U.M[1][:, 1:, :, :])[np.newaxis,:, :, :, :], 
                (U.M[2][:, :, :-1, :] + U.M[2][:, :, 1:, :])[np.newaxis,:, :, :, :],
                (U.M[3][:, :, :, :-1] + U.M[3][:, :, :, 1:])[np.newaxis,:, :, :, :]),axis=0)/2
    return V

def interp_adj(V):
    d = list(V.shape[1:])
    U = staggered(d)
    U.M[0] = np.concatenate((V[0, 0:1, :, :, :], V[0, :-1, :, :, :] + V[0, 1:, :, :, :], V[0, -1:, :, :, :]), axis=0) / 2
    U.M[1] = np.concatenate((V[1, :, 0:1, :, :], V[1, :, :-1, :, :] + V[1, :, 1:, :, :], V[1, :, -1:, :, :]), axis=1) / 2
    U.M[2] = np.concatenate((V[2, :, :, 0:1, :], V[2, :, :, :-1, :] + V[2, :, :, 1:, :], V[2, :, :, -1:, :]), axis=2) / 2
    U.M[3] = np.concatenate((V[3, :, :, :, 0:1], V[3, :, :, :, :-1] + V[3, :, :, :, 1:], V[3, :, :, :, -1:]), axis=3) / 2
    return U

def zero_boundary(U):
    tmp = staggered(U.dim)
    tmp.M[0] = U.M[0].copy()
    tmp.M[1] = U.M[1].copy()
    tmp.M[2] = U.M[2].copy()
    tmp.M[3] = U.M[3].copy()
    tmp.M[0][0,:,:,:] = 0
    tmp.M[0][-1, :, :,:] = 0
    tmp.M[1][:,0,:,:] = 0
    tmp.M[1][:,-1,:,:] = 0
    tmp.M[2][:,:,0,:] = 0
    tmp.M[2][:,:,-1,:] = 0
    tmp.M[3][:,:,:,0] = 0
    tmp.M[3][:,:,:,-1] = 0
    return tmp

def multi(a, U):
    tmp = staggered(U.dim)
    tmp.M[0] = U.M[0] * a
    tmp.M[1] = U.M[1] * a
    tmp.M[2] = U.M[2] * a
    tmp.M[3] = U.M[3] * a
    return tmp

def minus(U1, U2):
    tmp = staggered(U1.dim)
    tmp.M[0] = U1.M[0] - U2.M[0]
    tmp.M[1] = U1.M[1] - U2.M[1]
    tmp.M[2] = U1.M[2] - U2.M[2]
    tmp.M[3] = U1.M[3] - U2.M[3]
    return tmp

def add(U1, U2):
    tmp = staggered(U1.dim)
    tmp.M[0] = U1.M[0] + U2.M[0]
    tmp.M[1] = U1.M[1] + U2.M[1]
    tmp.M[2] = U1.M[2] + U2.M[2]
    tmp.M[3] = U1.M[3] + U2.M[3]
    return tmp

def div(U):
    d = U.dim
    temp = 0
    for k in range(len(d)):
        temp += np.diff(U.M[k], axis=k) * d[k]
    return temp

def ProxF(V, gamma, epsilon):
    '''compute Prox_2gamma F'''
    vs = V.shape
    m_1 = np.reshape(V[1, :, :, :, :], (np.prod(vs[1:]),1))
    m_2 = np.reshape(V[2, :, :, :, :], (np.prod(vs[1:]),1))
    m_3 = np.reshape(V[3, :, :, :, :], (np.prod(vs[1:]),1))
    f = np.reshape(V[0, :, :, :, :], (np.prod(vs[1:]), 1))
    
    P = np.concatenate((np.ones((len(f),1)), 4*gamma-f, 4*gamma**2-4*gamma*f, -4*gamma**2*f-gamma*(m_1**2 + m_2**2 + m_3**2)), axis=1)
    f = cubicpolynomial_maxrealsolution(P)
    
    I = f < epsilon
    f[I] = 0
    f = f.reshape((np.prod(vs[1:]),1))
    m_1 = ((m_1 * f) / (f+2*gamma)).reshape(vs[1:])
    m_2 = ((m_2 * f) / (f+2*gamma)).reshape(vs[1:])
    m_3 = ((m_3 * f) / (f+2*gamma)).reshape(vs[1:])
    f = f.reshape(vs[1:])
    
    return np.concatenate((f[np.newaxis,:, :, :, :], m_1[np.newaxis,:, :, :, :], m_2[np.newaxis,:, :, :, :], m_3[np.newaxis, :, :, :, :]), axis=0)

def cubicpolynomial_maxrealsolution(P):
    Z = np.zeros((len(P)))
    j = cmath.exp(2 * 1j * math.pi / 3)
    p = P[:,2] - P[:,1]**2/3
    q = 2 * P[:,1]**3 / 27 - P[:,1]*P[:,2]/3 + P[:,3]
    delta = q**2 + p**3*4/27
    
    Ip = np.where(delta > 0)
    Z[Ip] = np.cbrt((-q[Ip]+np.sqrt(delta[Ip])) / 2) + np.cbrt((-q[Ip]-np.sqrt(delta[Ip])) / 2)
    
    In = np.where(delta < 0)
    u = ((-q[In] + 1j * np.sqrt(-delta[In])) / 2) ** (1/3)
    z_1 = (u + np.conj(u)).real
    z_2 = (j * u + np.conj(j * u)).real
    z_3 = (j**2 * u + np.conj(j**2 * u)).real
    Z[In] = np.max((z_1, z_2, z_3),axis=0)
    
    Iz1 = np.where((delta == 0) & (p == 0))
    Z[Iz1] = 0
    
    Iz2 = np.where((delta == 0) & (p != 0))
    t_1 = 3*q[Iz2]/p[Iz2]
    t_2 = -3*q[Iz2] / (2*p[Iz2])
    Z[Iz2] = np.max((t_1,t_2),axis=0)
    
    return Z - P[:,1]/3

def ProxFS(V, sigma, epsilon):
    '''compute Prox_2sigma F^*'''
    return V - sigma * ProxF(V/sigma, 1/sigma, epsilon)

def Proj(U):
    t = 10**(-8)
    if np.sqrt(np.sum(U.M[1][:,0,:,:]**2))>t or np.sqrt(np.sum(U.M[1][:,-1,:,:]**2))>t or \
    np.sqrt(np.sum(U.M[2][:,:,0,:]**2))>t or np.sqrt(np.sum(U.M[2][:,:,-1,:]**2))>t or \
    np.sqrt(np.sum(U.M[3][:,:,:,0]**2))>t or np.sqrt(np.sum(U.M[3][:,:,:,-1]**2))>t or \
    np.abs(np.sum(U.M[0][0,:,:,:])-1) > t or np.abs(np.sum(U.M[0][-1,:,:,:])-1) > t:
        warnings.warn('Projection problem')
    
    d = U.dim
    dU = div(U) 
    p = poisson4d_Neumann(-dU)
    U.M[1][:,1:-1,:, :] += np.diff(p, axis=1)*d[1]
    U.M[2][:,:,1:-1, :] += np.diff(p, axis=2)*d[2]
    U.M[3][:,:,:, 1:-1] += np.diff(p, axis=3)*d[3]
    U.M[0][1:-1,:,:] += np.diff(p, axis=0)*d[0]
    
#     if np.min(U.M[0])<0:
#         if np.min(U.M[0])>-10**(-4):
#             I = U.M[0] < epsilon
#             U.M[0][I] = epsilon
#         else:
#             warnings.warn('Projection problem')    
    
    return U

def poisson4d_Neumann(f):
    '''Solve 4D Poisson equation with Neumann boundary conditions'''
    Q, N1, N2, N3 = f.shape
    
    h0 = 1 / Q
    h1 = 1 / N1
    h2 = 1 / N2
    h3 = 1 / N3
    
    dn = np.arange(Q)
    dep0 = 2 * np.cos(np.pi * dn / Q) - 2
    dm = np.arange(N1)
    dep1 = 2 * np.cos(np.pi * dm / N1) - 2
    dr = np.arange(N2)
    dep2 = 2 * np.cos(np.pi * dr / N2) - 2
    d3 = np.arange(N3)
    dep3 = 2 * np.cos(np.pi * d3 / N3) - 2
    
    A = np.repeat((dep0 / h0**2).reshape(Q,1), N1, axis=1)
    A = np.repeat(A[:,:,np.newaxis], N2, axis=2)
    A = np.repeat(A[:,:,:,np.newaxis], N3, axis=3)
    
    B = np.repeat((dep1 / h1**2).reshape(N1,1), N2, axis=1)
    B = np.repeat(B[np.newaxis, :,:], Q, axis=0)
    B = np.repeat(B[:, :,:, np.newaxis], N3, axis=3)
    
    C = np.repeat((dep2 / h2**2).reshape(N2,1), N3, axis=1)
    C = np.repeat(C[np.newaxis,:,:], N1, axis=0)
    C = np.repeat(C[np.newaxis,:,:], Q, axis=0)
    
    D = np.repeat((dep3 / h3**2)[np.newaxis,:], N2, axis=0)
    D = np.repeat((D[np.newaxis,:,:]), N1, axis=0)
    D = np.repeat((D[np.newaxis,:,:,:]), Q, axis=0)
    
    denom2 = A + B + C + D
    
    
    fhat = scipy.fftpack.dctn(f, type=2, shape=None, axes=None, norm='ortho', overwrite_x=False)
    denom2[denom2 == 0] = 1
    vhat = fhat / denom2
    res  = scipy.fftpack.idctn(vhat, type=2, shape=None, axes=None, norm='ortho', overwrite_x=False)
    
    return res

def train(f0, f1, Q, N1, N2, N3, iter_num, sigma, tau, theta, epsilon):
    t = np.tile(np.reshape(np.linspace(0, 1, Q + 1), (Q+1, 1, 1, 1)), (1, N1, N2, N3))
    f_init = (1-t) * np.repeat(f0[np.newaxis,:,:,:], Q+1, axis=0) + t * np.repeat(f1[np.newaxis, :, :, :], Q + 1, axis=0)
    d = [Q, N1, N2, N3]
    U = staggered(d)
    U.M[0] = f_init
    V = interp(zero_boundary(U))
    
    U_old = staggered(d)
    U_bar = staggered(d)
    U_bar.M = U.M.copy()
    
    for i in range(iter_num):
        
        U_old.M = U.M.copy()
        V = ProxFS(V + sigma*interp(U_bar),sigma, epsilon)
        U = Proj(minus(U, multi(tau, zero_boundary(interp_adj(V)))))
        U_bar = add(U, multi(theta, minus(U,U_old)))

    return U,V
