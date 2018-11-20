# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:37:11 2018

@author: bshrima2

Sparse Matrix as such not needed for the small dimensionality of the problem!
Solving a 1D BVP:
    
    -(EA(x)u'(x))'+Cu=T  for x in (0,L)
    u(0)=u(1)=0  : Homogeneous BC's

with quadratic Lagrange elements , 
now (hopefully) with quadratic (at least) p-Hierarchical elements
Parameters: 
    
    L  = Length of the domain
    A  = Area function A(x)




FIX ME: 
    change the exact solution for plotting 
    include p-hierarchical shape functions (FEM not GFEM)
    formulate the vandermonde matrix for approximating solution 
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as nla
from scipy.integrate import quad,solve_bvp
import scipy.sparse.linalg as sla
import scipy.sparse as sp
from scipy.interpolate import splrep,splev,interp1d
from matplotlib.ticker import AutoMinorLocator, LogLocator
from matplotlib import rcParams
from sympy import Symbol,diff,Array,lambdify,legendre,integrate,simplify,sin,pi,Matrix
import pandas as pd 
from mpmath import quadts as mpquads

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
#plt.rc('text',usetex=True)
plt.rcParams['xtick.top']='True'
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.right']='True'
plt.rcParams['ytick.direction']='in'
plt.rcParams['ytick.labelsize']=22
plt.rcParams['xtick.labelsize']=22
plt.rcParams['xtick.minor.visible']=True
plt.rcParams['ytick.minor.visible']=True
plt.rcParams['xtick.major.size']=6
plt.rcParams['xtick.minor.size']=3
plt.rcParams['ytick.major.size']=6
plt.rcParams['ytick.minor.size']=3

sptsy=np.array([i for i in np.linspace(0.1,1,20)])
sptsx=np.array([i for i in np.linspace(0.1,1,20)])

xmintick=AutoMinorLocator(20)
ymintick=AutoMinorLocator(20)
ymintickLog=LogLocator(base=10.0,subs=tuple(sptsy),numticks=len(sptsy)+1)
xmintickLog=LogLocator(base=10.0,subs=tuple(sptsx),numticks=len(sptsx)+1)



def ProblemBVP1(NumEl,El,MapX,enrch,numgauss):  
    
    class geometry() : #1D geometry parameters    
        def __init__(self,ne,deg):
            self.L=1.
            self.A=1.
            self.E=1.
            self.C=0.
            self.nLNodes=ne+1                                                      #Linear elements
            self.nQNodes=2*ne+1                                                    #Quadratic elements 
            self.nNNodes=int(ne*deg+1)                                             #Isoparametric deg-th order lagrange elements 
            self.ndim=2                                                            #No. of dofs per node (changes for GFEM isotropic enrichment)
            self.M=0
            self.g=0.
            self.xs = 0.
    
    def uex(x):
        return x**(0.65)+np.sin(3*np.pi*x/2.)
    
    class GPXi():
        def __init__(self,ordr):
            from numpy.polynomial.legendre import leggauss                         #Gauss-Legendre Quadrature for 1D
    #        from numpy.polynomial.chebyshev import chebgauss                      #Gauss-Chebyshev Quadrature for 1D (bad!)
            self.xi=leggauss(ordr)[0]
            self.wght=leggauss(ordr)[1]
    
    class basis():                                                                 # defined on the canonical element (1D : [-1,1] ) 
        def __init__(self,deg,basis_type):
            deg = int(deg)
            if basis_type == 'L':                                                  #1D Lagrange basis of degree deg
                z=Symbol('z',real=True)
                Xi=np.linspace(-1,1,deg+1)
                def lag_basis(k):
                    n = 1.
                    for i in range(len(Xi)):
                        if k != i:
                            n *= (z-Xi[i])/(Xi[k]-Xi[i])
                    return n
                N = Array([lag_basis(m) for m in range(deg+1)])            
                dfN = diff(N,z)+1.e-25*N
                self.Ns=lambdify(z,N,'numpy')
                self.dN=lambdify(z,dfN,'numpy')
                self.enrich=enrch
                ue = z**0.65+sin(3*pi*z/2)
                uep = ue.diff(z)
                uepp = uep.diff(z)
                self.Tf = lambdify(z,-uepp,'numpy')
                self.dE = float(integrate(0.5*uep**2,(z,0,1)).evalf(20))
            elif basis_type == 'M':                                                #1D Legendre Polynomials (Fischer calls this spectral)
                z = Symbol('z',real=True)
                x=Symbol('x')
                def gen_legendre_basis(n):
                    if n==-2:
                        return  (1-1*z)/2
                    elif n==-1:
                        return (1+1*z)/2
                    else:
                        return ((2*(n+3)-3)/2.)**0.5*integrate(legendre((n+1),x),(x,-1,z))
                N=Array([gen_legendre_basis(i-2) for i in range(deg+1)])
                N2 = N.tolist()
                N2[-1] = N[1]
                N2[1] = N[-1]
                N=Array(N2)
                dfN=diff(N,z)+1.e-25*N
    #            print(N)
                self.Ns=lambdify(z,N,'numpy')
                self.dN=lambdify(z,dfN,'numpy')
                self.enrich=enrch
                ue = z**0.65+sin(3*pi*z/2)
                uep = ue.diff(z)
                uepp = uep.diff(z)
                self.Tf = lambdify(z,-uepp,'numpy')
                self.dE = float(integrate(0.5*uep**2,(z,0,1)).evalf(20))
            elif basis_type == 'G':                                                #1D GFEM Functions (degree denotes the total approximation space)
                z=Symbol('z',real=True)
                Xi=np.linspace(-1,1,deg+1)
                def lag_basis(k):
                    n = 1.
                    for i in range(len(Xi)):
                        if k != i:
                            n *= (z-Xi[i])/(Xi[k]-Xi[i])
                    return n
                N = Array([simplify(lag_basis(m)) for m in range(deg+1)])            
                dfN = diff(N,z)+1.e-25*N
                self.Ns=lambdify(z,N,'numpy')
                self.dN=lambdify(z,dfN,'numpy')
                self.enrich=enrch                                                 #Enrichment order corresponding to GFEM shape functions 
                ue = z**0.65+sin(3*pi*z/2)
                uep = ue.diff(z)
                uepp = uep.diff(z)
                self.Tf = lambdify(z,-uepp,'numpy')
                self.dE = float(integrate(0.5*uep**2,(z,0,1)).evalf(20))
            elif basis_type == 'Singl' or basis_type == 'SGFEMSingl' or basis_type == 'GeomSingl' or basis_type == 'GeomSGFEM':
                z=Symbol('z',real=True)
                xalph=Symbol('xalph',real=True)
                halph=Symbol('halph',real=True)
                Xi=np.linspace(-1,1,deg+1)
                def lag_basis(k):
                    n = 1.
                    for i in range(len(Xi)):
                        if k != i:
                            n *= (z-Xi[i])/(Xi[k]-Xi[i])
                    return n
                N = Array([simplify(lag_basis(m)) for m in range(deg+1)])            
                dfN = diff(N,z)+1.e-27*N
                
#                Manufactured solution
                
                ue = z**0.65+sin(3*pi*z/2)
                uep = ue.diff(z)
                uepp = uep.diff(z)
                Nf2 = Array([(ue)**m * ((z-xalph)/halph)**n +1.e-27*(z-xalph)/halph for m in range(2) for n in range(enrch+1)])   # Enrichment by non-polynomials (for interfaces) 
                dfN2 = Nf2.diff(z) + 1.e-26*Nf2
                
#                Convert to numpy functions to do array operations (vectorized)
                self.Ns=lambdify(z,N,'numpy')
                self.dN=lambdify(z,dfN,'numpy')
                self.enrich=enrch     
                self.Tf = lambdify(z,-uepp,'numpy')
                self.dE = float(integrate(0.5*uep**2,(z,0,1)).evalf(20))
                self.uexact=lambdify(z,ue,'numpy')
                self.NIF=lambdify((z,xalph,halph),Nf2,'numpy')                 # Final enrichment functions including both polynomial and exact soln
                self.DNIF=lambdify((z,xalph,halph),dfN2,'numpy')
                self.enrich=enrch 
            else:
                raise Exception('Element type not implemented yet')
                
    def fexact(x,y):
        u,up=y
        return [up,1/(geom.E*geom.A)*(geom.C*u-B.Tf(x))]

    def fbc(ya,yb):
        return np.array([ya[0],yb[0]])
        
    def Ej(x,xalph,halph,enrich):
        z= Symbol('z')
        N = Array([((z-xalph)/halph)**n + 1.e-26*(z-xalph)/halph for n in range(enrich+1)])   #shape consistency take E_o = 1
        Nf = lambdify(z,N,'numpy')
        return Nf(x)
    
    def Ejp(x,xalph,halph,enrich):
        z= Symbol('z')
        N = Array([((z-xalph)/halph)**n + 1.e-26*(z-xalph)/halph for n in range(enrich+1)])   #shape consistency take E_o = 1
        dfN = lambdify(z,diff(N,z)+1.e-26*N,'numpy')                                 #shape consistency
        return dfN(x)                                                              
    
    
    def ge_singl_stiff_matrx(eflag,etypes,znodes):                                                # znodes = nodes of current elements        
        eabs=1.e-12
        limits=10000
        xi = Symbol('xi',real=True)
        z = Symbol('z',real=True)
        x = 0.5*( (1-xi)*znodes[0] + (1+xi)*znodes[-1]) 
        Je = znodes[-1]-znodes[0]
        Je /= 2.
#        print(Je)
        ue = x**0.65+sin(3*pi*x/2);uez = z**0.65+sin(3*pi*z/2)
        Tx = -1.*uez.diff(z,2)
        Tx_xi = Tx.subs([(z,0.5*((1-xi)*znodes[0] + (1+xi)*znodes[-1]))])
        phi1 = 0.5*(1-xi);phi2 = 0.5*(1+xi)
        if elflag == 'nn':
            Nshp = Array([phi1,phi2])
            dNshp = Nshp.diff(xi)
        elif elflag == 'repr' or elflag == 'bll':
            if etypes =='Singl':
                Nshp = Array([phi1,phi1*ue,phi2])
                dNshp = Nshp.diff(xi)
            
            elif etypes == 'SGFEMSingl':
                ue_interp = ue-phi1*uez.subs(z,znodes[0])-phi2*uez.subs(z,znodes[-1])
                Nshp = Array([phi1,phi1*ue_interp,phi2])
                dNshp = Nshp.diff(xi)
            
            elif etypes == 'GeomSGFEM':
                ue_interp = ue-phi1*uez.subs(z,znodes[0])-phi2*uez.subs(z,znodes[-1])
                if elflag == 'repr':
                    Nshp = Array([phi1,phi1*ue_interp,phi2,phi2*ue_interp])
                    dNshp = Nshp.diff(xi)
                elif elflag == 'bll':
                    Nshp = Array([phi1,phi1*ue_interp,phi2])
                    dNshp = Nshp.diff(xi)
            
            elif etypes == 'GeomSingl':
                ue_interp = ue-phi1*uez.subs(z,znodes[0])-phi2*uez.subs(z,znodes[-1])
                if elflag == 'repr':
                    Nshp = Array([phi1,phi1*ue,phi2,phi2*ue])
                    dNshp = Nshp.diff(xi)
                elif elflag == 'bll':
                    Nshp = Array([phi1,phi1*ue,phi2])
                    dNshp = Nshp.diff(xi) 
        
        stiffM = Matrix(dNshp)*Matrix(dNshp).T*geom.E*geom.A/Je + Matrix(Nshp)*Matrix(Nshp).T*geom.C*geom.A*Je
        forvec = Nshp*Tx_xi*Je
#        mat = np.array([[quad(lambdify(xi,stiffM[i,j],'numpy'),-1.,1.,points=[-1.])[0] for i in range(stiffM.shape[0])] for j in range(stiffM.shape[1])],float)
        mat = np.array([[quad(lambdify(xi,stiffM[i,j],'numpy'),-1.,1.,epsabs=eabs,limit=limits,points=[-1.])[0] for i in range(stiffM.shape[0])] for j in range(stiffM.shape[0])],float)
#        elemF = np.array([quad(lambdify(xi,forvec[i],'numpy'),-1.,1.,points=[-1.])[0] for i in range(len(forvec))],float)
        elemF = np.array([quad(lambdify(xi,forvec[i],'numpy'),-1.,1.,epsabs=eabs,limit=limits,points=[1.])[0] for i in range(len(forvec))],float)
        return mat,elemF.flatten() 
    
    def loc_mat(nodes,Elflag,Ndnode,etypes):                                         #Computes the element quantities when supplied with global nodes
        xi=GP.xi;W=GP.wght
    #    print(Je)
        if etypes=='L' or etypes=='M':                                               # 1D lagrange or legendre basis
            x=np.array(nB.Ns(xi)).T @ nodes
            Je=np.array(nB.dN(xi)).T @ nodes
            b1 = np.array(B.Ns(xi)).reshape(-1,1,W.size)                           
            a1 = np.array(B.dN(xi)).reshape(-1,1,W.size)  
            a2=a1.reshape(1,len(a1),-1).copy()
            b2=b1.reshape(1,len(b1),-1).copy()
            a1 *= geom.E/Je*geom.A*W                                               #multiply by weights, jacobian, A, in bitwise fashion 
            b1 *= geom.C*Je*W
            mat=np.tensordot(a1,a2,axes=([1,2],[0,2])) + np.tensordot(b1,b2,axes=([1,2],[0,2]))     #could potentially use einsum to clean it up (but works for now!)
            elemF=(np.array(B.Ns(xi))*W*Je*B.Tf(x)).sum(axis=1).flatten()
#            print('\n',mat)
            return mat,elemF
        elif etypes=='G':                                                          #GFEM approximation 
            halph = nodes[-1] - nodes[0]
            enrch = B.enrich                                                       #enrichment order 
            x=np.array(nB.Ns(xi)).T @ nodes
            Je=np.array(nB.dN(xi)).T @ nodes
            varphi_alph = np.array(B.Ns(xi)).reshape(1,-1,W.size)                  #\varphi_1 evaluated at gauss points (along 3rd Dimensn)
    #        print(varphi_alph.shape)
            varphi_alph_p = 1/Je* np.array(B.dN(xi)).reshape(1,-1,W.size)          #\varphi_1' evaluated at gauss points (along 3rd Dimensn)
            
            Ealph = np.array(Ej(x,nodes,halph,enrch))
    #        print(Ealph.shape)
            Ealphp = np.array(Ejp(x,nodes,halph,enrch))
    #        Ealphp[0]=1.                                                           #Gradients work slightly differently (cannot take bitwise product of gradient of individual functions)
    
            varphi_alph_E_alph = np.einsum('ijk,ijk->ijk',varphi_alph,Ealph)
    #        print(varphi_alph_E_alph[0,1,:])
            Nf = np.einsum('jik',varphi_alph_E_alph).reshape(-1,1,W.size)
    #        print(Nf[2,0,:])
            
            varphi_alph_E_alph_p = np.einsum('ijk,ijk->ijk',varphi_alph_p,Ealph) + np.einsum('ijk,ijk->ijk',varphi_alph,Ealphp)      
            dNf = np.einsum('jik',varphi_alph_E_alph_p).reshape(-1,1,W.size)
            
            mat = geom.E*geom.A*W*Je*np.einsum('ikm,jkm->ijm',dNf,dNf)
            mat = mat.sum(axis=-1)
    #        print(mat.shape)
            M_mat = geom.C*geom.A*W*Je*np.einsum('ikm,jkm->ijm',Nf,Nf) 
            M_mat = M_mat.sum(axis=-1)
            mat += M_mat                                                                # Taking into account any terms with C(x)
            elemF = (W*Je*Nf*B.Tf(x)).sum(axis=-1)
            return mat,elemF.flatten()
        
        elif etypes == 'Singl' or etypes == 'SGFEMSingl':
            halph = nodes[-1] - nodes[0]
            xi=GP.xi;W=GP.wght
            x=np.array(B.Ns(xi)).T @ nodes
            Je=np.array(B.dN(xi)).T @ nodes           
            
            if Elflag != 'nn':
                mat,elemF = ge_singl_stiff_matrx(Elflag,etypes,nodes)
            elif Elflag == 'nn':
                mat,elemF = ge_singl_stiff_matrx(Elflag,etypes,nodes)
#                b1 = np.array(B.Ns(xi)).reshape(-1,1,W.size)                           
#                a1 = np.array(B.dN(xi)).reshape(-1,1,W.size)  
#                a2=a1.reshape(1,len(a1),-1).copy()
#                b2=b1.reshape(1,len(b1),-1).copy()
#                a1 *= geom.E/Je*geom.A*W                                               #multiply by weights, jacobian, A, in bitwise fashion 
#                b1 *= geom.C*Je*W
#                mat=np.tensordot(a1,a2,axes=([1,2],[0,2])) + np.tensordot(b1,b2,axes=([1,2],[0,2]))     #could potentially use einsum to clean it up (but works for now!)
#                elemF=(np.array(B.Ns(xi))*W*Je*B.Tf(x)).sum(axis=1).flatten()
            return mat,elemF
        
        elif etypes == 'GeomSGFEM' or etypes == 'GeomSingl':
            halph = nodes[-1] - nodes[0]
            xi=GP.xi;W=GP.wght
            x=np.array(B.Ns(xi)).T @ nodes
            Je=np.array(B.dN(xi)).T @ nodes
            if Elflag != 'nn':
                mat,elemF = ge_singl_stiff_matrx(Elflag,etypes,nodes)
            elif Elflag == 'nn':
                mat,elemF = ge_singl_stiff_matrx(Elflag,etypes,nodes)
#                b1 = np.array(B.Ns(xi)).reshape(-1,1,W.size)                           
#                a1 = np.array(B.dN(xi)).reshape(-1,1,W.size)  
#                a2=a1.reshape(1,len(a1),-1).copy()
#                b2=b1.reshape(1,len(b1),-1).copy()
#                a1 *= geom.E/Je*geom.A*W                                               #multiply by weights, jacobian, A, in bitwise fashion 
#                b1 *= geom.C*Je*W
#                mat=np.tensordot(a1,a2,axes=([1,2],[0,2])) + np.tensordot(b1,b2,axes=([1,2],[0,2]))     #could potentially use einsum to clean it up (but works for now!)
#                elemF=(np.array(B.Ns(xi))*W*Je*B.Tf(x)).sum(axis=1).flatten()
            return mat,elemF
                
            
    def fsample(x,dof,nodes,sample_type):                                          #giving out the displacement, strain and stress at x: GP and dof value of disp. at gauss point
    #    print(sample_type)
        if sample_type=='spline':
            if int(El[1]) >5:
                k0=5
            else:
                k0 = int(El[1])
                dspl=splrep(nodes,dof,k=k0)
                return (x,splev(x,dspl))
        elif sample_type=='vandermonde':
            Xinew = np.linspace(-1,1-1.e-13,20)
            N = np.array(B.Ns(Xinew)).reshape(int(El[-1])+1,Xinew.size,-1)
            N = np.einsum('jik',N)
            Nx = np.array(nB.Ns(Xinew)).reshape(int(MapX[-1])+1,Xinew.size,-1)
            Nx = np.einsum('jik',Nx)
            dofarang = np.vstack((np.arange(k,k+dof.size-1,int(El[-1])) for k in range(int(El[-1])+1))) #arange dofs for multiplication with n
            dofarang=dofarang.reshape(int(El[-1])+1,1,-1)
            xarang = np.vstack((np.arange(k,k+nodes.size-1,int(MapX[-1])) for k in range(int(MapX[-1])+1)))
            xarang = xarang.reshape(int(MapX[-1])+1,1,-1)
    #        print(dof.shape)
    #        print(nodes.shape)
            dofs=dof[dofarang]
            nodesx=nodes[xarang]
            soln = np.einsum('ikj,kmj->imj',N,dofs)
            xpl = np.einsum('ikj,kmj->imj',Nx,nodesx)
            return (xpl.T.ravel(),soln.T.ravel())                                  # return the solution evaluated at the given points 
    
    def babuskasolve(K,f):
        T=np.diag(np.diag(K)**(-0.5))
        Keps=T @ K @ T + 1.e-15*np.eye(len(K))                                     # Remove the zero eigenvalues of the stiffness matrix, after static condensation
        feps = T @ f
        u0 = nla.solve(Keps,feps)
        r0 = feps - Keps @ u0
        e0 = nla.solve(Keps,r0)
        e=e0.copy();r=r0.copy();u=u0.copy()
        res=abs(e @ (Keps @ e)/(u @ (Keps @ u)))
        iters=0
        
        while res >= 1.e-12:
            iters += 1
            r -= Keps @ e
            e = nla.solve(Keps,r)
            u += e
            res = abs(e @ (Keps @ e)/(u @ (Keps @ u)))
            print('i=',iters)
#        print(nla.cond(Keps))
        return T @ u    
    
    etypes="".join(El.rsplit(El[-1]))
    Np = numgauss                                                                        #Order of Gauss-Integration for the Discrete Galerkin-Form 
    Nel=NumEl                                                                      #Ask why energy is coming out higher in single precision?
    
    if etypes=='M' or etypes=='L':                                                 #1D Mapping function for the physical coordinates (can take only single digits, some modification needed)
        nB=basis(float(MapX[-1]),MapX[0])
        geom=geometry(Nel,float(MapX[-1])) 
        probsize=geometry(Nel,float(El[-1]))
    elif etypes=='G' or etypes=='Singl' or etypes == 'SGFEMSingl' or etypes == 'GeomSingl' or etypes == 'GeomSGFEM':
        nB=basis(float(El[-1]),etypes)
        geom=geometry(Nel,float(MapX[-1])) 
        probsize=geometry(Nel,float(El[-1]))
    B=basis(float(El[-1]),etypes)                                                   #Basis for FE fields (isoparametric) 
#    T = B.Tf
    GP=GPXi(Np) 
    
    if etypes=='M' or etypes=='L':
        nodes = np.linspace(geom.xs,geom.L,geom.nNNodes)
        elems = np.vstack((np.arange(k,k+nodes.size-1,int(MapX[-1])) for k in range(int(MapX[-1])+1))).T
        globK=0*np.eye((B.enrich)*probsize.nNNodes)                                # This changes because now we no longer have iso-p map
        globF=np.zeros((B.enrich)*probsize.nNNodes)    
        prescribed_dof=np.array([[0,0],
                                 [-B.enrich,0]])
    elif etypes=='G':
        nodes = np.linspace(geom.xs,probsize.L,probsize.nNNodes)
        elems = np.vstack((np.arange(k,k+nodes.size-1,int(El[-1])) for k in range(int(El[-1])+1))).T
        globK=0*np.eye((B.enrich+1)*probsize.nNNodes)                              # This changes because now we no longer have iso-p map
        globF=np.zeros((B.enrich+1)*probsize.nNNodes) 
        prescribed_dof=np.array([[0,0],
                                 [-B.enrich-1,0]])

    elif etypes=='Singl' or etypes == 'SGFEMSingl':
        nodes = np.linspace(geom.xs,probsize.L,probsize.nNNodes)
        elems = np.vstack((np.arange(k,k+nodes.size-1,int(El[-1])) for k in range(int(El[-1])+1))).T
        globK=0*np.eye((B.enrich)*probsize.nNNodes+1)                                # Accounting for an additional B.enrich+1 dofs due to the singularity at one node
        globF=np.zeros((B.enrich)*probsize.nNNodes+1) 
        prescribed_dof=np.array([[0,0],
                                 [-B.enrich,0]])                                 # Remains the same      

    elif etypes == 'GeomSGFEM' or etypes == 'GeomSingl':
        nodes = np.linspace(geom.xs,probsize.L,probsize.nNNodes)
        elems = np.vstack((np.arange(k,k+nodes.size-1,int(El[-1])) for k in range(int(El[-1])+1))).T
        nodes_enriched_idx = nodes <= 0.25*geom.L
        num_nodes_enriched = int(nodes_enriched_idx.sum())
#        enriched_nodes = nodes[nodes_enriched]
        num_elems_enriched = int(num_nodes_enriched - 1)
        globK = 0*np.eye((B.enrich+1)*num_nodes_enriched+(probsize.nNNodes - num_nodes_enriched))
        globF = np.zeros(globK.shape[0])
        prescribed_dof=np.array([[0,0],
                                 [-B.enrich,0]])                                 # Remains the same 
#    print('\nSinglStiff: = \n',ge_singl_stiff_matrx(nodes[[0,1]])[0])
    
    dof=np.inf*np.ones(len(globK))                                                 # initialize to infinity 
    
    prescribed_forc=0*np.array([[-1,-geom.M*geom.g]])                              # Don't use `int` here! (prescribed concentrated loads) 
    
    dof[prescribed_dof[:,0]]=prescribed_dof[:,1]
    
    for k in range(elems[:,0].size):                                               #Assembly of Global System
        elflag = 'nn'
        ndnode = 1
        elnodes=elems[k]
        if etypes=='M' or etypes=='L':
            elemord=int(El[-1])+1
            globdof = np.array([k*(elemord-1)+i for i in range(B.enrich*elemord)],int)
        elif etypes=='G':
            elemord = B.enrich+int(El[-1])
            ndofsel=(int(El[-1])+1)*(B.enrich+1)
            globdof = np.array([k*(ndofsel-B.enrich-1)+i for i in range(ndofsel)],int)
    #        globdof = globdof[[0,2,3,1]]
    
        elif etypes=='Singl' or etypes=='SGFEMSingl':
            elemord = B.enrich+int(El[-1])
            ndofsel=(int(El[-1])+1)*(elemord)
            if k==0:
                elflag='bll'
#                globdof = [0,1,2]
                globdof = np.array([k*(ndofsel-elemord)+i for i in range(ndofsel-1)],int)
            else:
                elflag = 'nn'
                elemord=int(El[-1])+1
                globdof = np.array([k*(elemord-1)+i for i in range(B.enrich*elemord)],int)+1
        
        elif etypes == 'GeomSingl' or etypes == 'GeomSGFEM': 
            elemord = B.enrich+int(El[-1])
            ndofsel=(int(El[-1])+1)*(elemord)
            if k <= num_elems_enriched-1:
                elflag = 'repr'
                globdof = np.array([k*(ndofsel-elemord)+i for i in range(ndofsel)],int)
            elif k == num_elems_enriched:
                elflag = 'bll'
                globdof = np.array([k*(ndofsel-elemord)+i for i in range(ndofsel-1)],int)
            else: 
                elflag = 'nn'
                globdof = np.array([k*(elemord-1)+i for i in range(B.enrich*elemord)],int)+num_nodes_enriched
        
        if etypes != '2D':                                                         # 1D lagrange or legendre polynomials 
            nodexy=nodes[elnodes]
        else:
            nodexy=nodexy[:,elnodes]
        kel,fel=loc_mat(nodexy,elflag,ndnode,etypes)
#        print('dof id = ',globdof)
#        print('elnodes = ',elnodes)
        
        globK[np.ix_(globdof,globdof)] += kel                                      #this process would change in vectorization 
        globF[globdof] += fel
#    print('index of nodes enriched = ',nodes_enriched_idx)
    globF[int(prescribed_forc[:,0])] += prescribed_forc[:,1]
    fdof=dof==np.inf                                                               # free dofs
    nfdof=np.invert(fdof)   
#    print(g)
    if etypes=='L' or etypes=='M':  
        AA=globK[np.ix_(fdof,fdof)]
        bb=globF[fdof]-globK[np.ix_(fdof,nfdof)] @ dof[nfdof]                                                                           
        dofme=nla.solve(AA,bb)
    elif etypes=='G' or etypes=='Singl' or etypes == 'SGFEMSingl' or etypes == 'GeomSingl' or etypes == 'GeomSGFEM':
        AA=globK[np.ix_(fdof,fdof)]
        bb=globF[fdof]-globK[np.ix_(fdof,nfdof)] @ dof[nfdof]
        dofme = babuskasolve(AA,bb)                                                #Use Bi-Conjugate Gradient to Solve Ax=b (?)
    dof[fdof]=dofme.copy()

    xplot=np.linspace(0.,geom.L,10*geom.nNNodes-1)                                              #sampling points for the exact solution
    xplot1 = xplot.copy()
    xplot1[0] += 1.e-17
    #plotfigs(xplot) 
    ue=uex(xplot)
    xpl2 = np.linspace(nodes[0],nodes[-1],10*geom.nQNodes-1)
    fittype='vandermonde'
    if Nel >= 20:
        plot_it=False
    else:
        plot_it = False
    if plot_it:
        if etypes=='L' or etypes=='M':
            if fittype=='spline':
                xpl2,uapp=fsample(xpl2,dof[0:len(dof):int(El[1])],nodes[0:len(nodes):2],fittype)
            elif fittype=='vandermonde':
                xpl2,uapp=fsample(xpl2,dof,nodes,fittype)
                
            y_a=np.empty((2,nodes.size))
            nodesSP = nodes.copy()
            nodesSP[0] += 1.e-17
            res_a=solve_bvp(fexact,fbc,nodesSP,y_a)
            
            plt.figure(figsize=(10,10))
            ax=plt.gca()
            ax.yaxis.set_minor_locator(ymintick)
            ax.xaxis.set_minor_locator(xmintick)
            
#            plt.text(0.35,0.33,r'$a=$ '+str(geom.a),transform=ax.transAxes,
#                     fontsize=22,bbox=dict(facecolor='blue', alpha=0.5))
            if etypes=='L':
                plt.plot(xpl2,uapp,'-o',label=r'Lagrange FE: degree = {0:1d}'.format(int(El[-1])))
            elif etypes=='M' or etypes=='G':
                plt.plot(xpl2,uapp,'-o',label=r'p-Hierarchical FEM: Degree '+str(El[1]))
            plt.plot(xplot,ue,'-o',label=r'Exact Solution')
            plt.plot(xplot1,res_a.sol(xplot1)[0],label=r'Scipy--solve bvp')
            
            plt.tick_params(axis='both',which='both',direction='in',top=True,right=True,labelsize=20)
            plt.title(r'Number of elements: '+str(Nel)+', NGP = '+str(Np),fontsize=20)
            plt.legend(loc=0,fontsize=18)
            ax.tick_params(which='major',length=6)
            ax.tick_params(which='minor',length=3)
            
            plt.grid(True,linestyle='--')
            plt.text(0.25,0.2,r'$U^h$ = '+str(0.5*dof @ (globK @ dof)),transform=ax.transAxes,
                     fontsize=22,bbox=dict(facecolor='red', alpha=0.5))
    
    Uexact = B.dE
    Uh = 0.5*dof @ (globK @ dof)
    errUh = np.sqrt(abs(Uh-Uexact)/Uexact)
#    print(errUh)
#    print('Uapp = ',Uh)
    A = globK[np.ix_(fdof,fdof)]
    K_mod = np.diag(np.diag(A)**(-0.5)) @ A @ np.diag(np.diag(A)**(-0.5))
#    print(nla.cond(K_mod))
#    print(nla.cond(A))
    return Uh,errUh,dof.size,globK,globF,nla.cond(K_mod)

if __name__ == '__main__':
#    xbval=0.2
#    aVal=0.5
    NGP = 90
    numels=np.array([2**(i) for i in range(3,10)],int)
    MapType='L1'
    enrch=1
    errs=np.zeros(len(numels))
    dofsarr=errs.copy()
    errsSG=np.zeros(len(numels))
    condSG = errs.copy()
    condS = errs.copy()
    dofsarrSG=errs.copy()
    dofsFEM=errs.copy()
    condFEM = errs.copy()
    errFEM=errs.copy()
    errGeom = errs.copy()
    dofsGeom = errs.copy()
    dofsGeomSGFEM = errs.copy()
    errGeomSGFEM = errs.copy()
    condGeom = errs.copy()
    condGeomSGFEM = errs.copy()
    ElTypes=['SGFEMSingl1','Singl1','L1','GeomSingl1','GeomSGFEM1']
#    ElTypes = ['GeomSingl1']
    for eltypes in ElTypes:
        for idx,ne in enumerate(numels):
            Uh,errUh,Nsize,SStiff,Fforc,condKmod = ProblemBVP1(ne,eltypes,MapType,enrch,NGP)
            if eltypes == 'SGFEMSingl1':
                errsSG[idx] = errUh
                dofsarrSG[idx] = Nsize
                condSG[idx] = condKmod
            elif eltypes == 'Singl1':
                errs[idx] = errUh
                dofsarr[idx] = Nsize
                condS[idx] = condKmod
            elif eltypes == 'L1':
                errFEM[idx] = errUh
                dofsFEM[idx] = Nsize
                condFEM[idx] = condKmod
            elif eltypes == 'GeomSingl1':
                errGeom[idx] = errUh
                dofsGeom[idx] = Nsize
                condGeom[idx] = condKmod
            elif eltypes == 'GeomSGFEM1':
                errGeomSGFEM[idx] = errUh
                dofsGeomSGFEM[idx] = Nsize
                condGeomSGFEM[idx] = condKmod
    
    plt.figure(figsize=(8,8))
    plt.loglog(dofsarr,errs,'-o',label=r'GFEM')
    plt.loglog(dofsarrSG,errsSG,'-o',label=r'SGFEM')
    plt.loglog(dofsFEM,errFEM,'-o',label=r'FEM')
    plt.loglog(dofsGeom,errGeom,'-o',label=r'Geometric-GFEM')
    plt.loglog(dofsGeomSGFEM,errGeomSGFEM,'-o',label=r'Geometric-SGFEM')
    ax=plt.gca()
    ax.yaxis.set_minor_locator(ymintickLog)
    ax.xaxis.set_minor_locator(xmintickLog)
    ax.set_xlabel(r'\# of dofs ($N$)',fontsize=22)
    ax.set_ylabel(r'Relative error in the energy norm: $\| e \|_{\mathcal{E}}$',fontsize=22)
    ax.legend(loc=0,fontsize=22)
    ax.tick_params(which='minor',labelleft=False,labelbottom=False)
    plt.grid(True,linestyle='--')
    plt.tight_layout()

    
    plt.figure(figsize=(8,8))
    plt.loglog(dofsarr,condS,'-o',label=r'GFEM')
    plt.loglog(dofsarrSG,condSG,'-o',label=r'SGFEM')
    plt.loglog(dofsFEM,condFEM,'-o',label=r'FEM')
    plt.loglog(dofsGeom,condGeom,'-o',label=r'Geometric-GFEM')
    plt.loglog(dofsGeomSGFEM,condGeomSGFEM,'-o',label=r'Geometric-SGFEM')
    ax=plt.gca()
    ax.yaxis.set_minor_locator(ymintickLog)
    ax.xaxis.set_minor_locator(xmintickLog)
    ax.tick_params(which='minor',labelleft=False,labelbottom=False)
    ax.set_xlabel(r'\# of dofs ($N$)',fontsize=22)
    ax.set_ylabel(r'Condition number: $\kappa (\widehat{K})$',fontsize=22)
    ax.legend(loc=0,fontsize=22)
    plt.grid(True,linestyle='--')
    plt.tight_layout()

    btaFEM = abs(np.log10(errFEM[-1]/errFEM[-2])/np.log10(dofsFEM[-1]/dofsFEM[-2]))
    btaGFEM = abs(np.log10(errs[-1]/errs[-2])/np.log10(dofsarr[-1]/dofsarr[-2]))
    btaSGFEM = abs(np.log10(errsSG[-1]/errsSG[-2])/np.log10(dofsarrSG[-1]/dofsarrSG[-2]))
    btaGFEMGeom = abs(np.log10(errGeom[-1]/errGeom[-2])/np.log10(dofsGeom[-1]/dofsGeom[-2]))
    btaSGFEMGeom = abs(np.log10(errGeomSGFEM[-1]/errGeomSGFEM[-2])/np.log10(dofsGeomSGFEM[-1]/dofsGeomSGFEM[-2])) 
    
    alphFEM = abs(np.log10(condFEM[-1]/condFEM[-2])/np.log10(dofsFEM[-1]/dofsFEM[-2]))
    alphGFEM = abs(np.log10(condS[-1]/condS[-2])/np.log10(dofsarr[-1]/dofsarr[-2]))
    alphSGFEM = abs(np.log10(condSG[-1]/condSG[-2])/np.log10(dofsarrSG[-1]/dofsarrSG[-2]))
    alphGFEMGeom = abs(np.log10(condGeom[-1]/condGeom[-2])/np.log10(dofsGeom[-1]/dofsGeom[-2]))
    alphSGFEMGeom = abs(np.log10(condGeomSGFEM[-1]/condGeomSGFEM[-2])/np.log10(dofsGeomSGFEM[-1]/dofsGeomSGFEM[-2]))
    
    # Writing the data points for tables 
    
    
