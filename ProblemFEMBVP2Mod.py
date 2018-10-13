# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:37:11 2018

@author: bshrima2

Sparse Matrix as such not needed for the small dimensionality of the problem!
Solving a 1D BVP:
    
    -(EA(x)u'(x))'+Cu=T  for x in (0,L)
    u(0)=0; u(L)=1  : Homogeneous BC's

Now with material interfaces and enrichment functions to accurately approximate 
the material interface 


FIX ME: 
    Add Sukumar enrichment which is the sign distance function (aid sympy)
    Add a condition to check for odd number of elements 
        - Quadrature has to be applied on integration sub-elements
        - Even taking arbitrary points won't work (?) - No exponential converg.
    Add additional dofs of the blending element (Line 278) which correpsond to 
    the correction function (since it requires enriching all the dofs of the 
    blending elements)
    
    
    
    
    
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as nla
from scipy.integrate import quad,solve_bvp
import scipy.sparse.linalg as sla
import scipy.sparse as sp
from scipy.interpolate import splrep,splev,interp1d
from sympy import Symbol, Array, diff, integrate, simplify, lambdify, legendre

plt.rc('text',usetex=True)

enrch=0

class geometry() : #1D geometry parameters    
    def __init__(self,ne,deg):
        self.L=1
        self.A=1.
        self.E1=1
        self.E2=2*self.E1
#        self.E2=10**3
        self.C=0.
#        self.a=50
        self.xgam=0.5*self.L
        self.nLNodes=ne+1       #Linear elements
        self.nQNodes=2*ne+1     #Quadratic elements 
        self.nNNodes=int(ne*deg+1)   #Isoparametric deg-th order lagrange elements 
        self.ndim=1
        self.M=0
        self.g=0.

class GPXi():
    def __init__(self,ordr):
        from numpy.polynomial.legendre import leggauss                         #Gauss-Legendre Quadrature for 1D
#        from numpy.polynomial.chebyshev import chebgauss                      #Gauss-Chebyshev Quadrature for 1D (bad!)
        self.xi=leggauss(ordr)[0]
        self.wght=leggauss(ordr)[1]

class basis():                                                                 # defined on the canonical element (1D : [-1,1] ) 
    def __init__(self,deg,basis_type):
        deg = int(deg)
        if basis_type == 'L':                                                  # 1D Lagrange basis of degree deg
            z=Symbol('z')
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
            self.enrich=1
        elif basis_type == 'M':                                                # 1D Legendre Polynomials (Fischer calls this spectral)
            z = Symbol('z')
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
            self.enrich=1
        elif basis_type == 'G':                                                #1D GFEM Functions (degree denotes the total approximation space)
            z=Symbol('z')
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
        else:
            raise Exception('Element type not implemented yet')
       
def E(x):
    Ef = geom.E2*np.heaviside(x-geom.xgam,0) + geom.E1*(1-np.heaviside(x-geom.xgam,0))
    return Ef

def g(x):
    return -25./6*x**3+5./8*x**4-1./40*x**5

def u1(x):
    return 1/geom.E1*(g(x)+geom.E2*x*Bp)

def u2(x):
    return Bp*(x-geom.L)+1.+1./geom.E2*(g(x)-g(geom.L))

def uex(x):
    return np.heaviside(x-geom.xgam,1.0)*u2(x)+(1-np.heaviside(x-geom.xgam,0))*u1(x)

def T(x):
    return 0*(25*x-7.5*x**2+0.5*x**3)

def fexact(x,y):
    u,up=y
    return [up,1/(E(x)*geom.A)*(geom.C*u-T(x))]

def fbc(ya,yb):
    return np.array([ya[0],yb[0]-1.])


def Lj(x,nodes):                                                               #Nodal Enrichment function Lj(x) for an element with nodes 'nodes'
    hL = abs(nodes[0]-geom.xgam)
    hR = abs(nodes[-1]-geom.xgam)
    return (x - geom.xgam)/(max(hL,hR))

def find_nodes(nodes):
    return abs(nodes-geom.xgam) <= geom.L/Nel

def Ej(x,xalph,halph,enrich):
    z= Symbol('z')
    N = Array([((z-xalph)/halph)**n + 1.e-26*(z-xalph)/halph for n in range(enrich+1)])   #shape consistency take E_o = 1
    Nf = lambdify(z,N,'numpy')
    return Nf(x)

def Ejp(x,xalph,halph,enrich):
    z= Symbol('z')
    N = Array([((z-xalph)/halph)**n + 1.e-26*(z-xalph)/halph for n in range(enrich+1)])   #shape consistency take E_o = 1
    dfN = lambdify(z,diff(N,z)+1.e-26*N,'numpy')                               #shape consistency
    return dfN(x)        

def Etilj(x,xgam):                                                       # Absolute signed distance function with smoothening 
    return np.heaviside(x-xgam,0.)*(x-xgam)/(geom.L-xgam)                         # Test for ramp function  
#    return abs(abs(x-xgam)*np.sign((x-xgam)))/np.max(abs(xalph-xgam))

def Etiljp(x,xgam):
    return np.heaviside(x-xgam,0.)/(geom.L-xgam)
#    return np.sign(x-xgam)*np.sign((x-xgam)*np.sign(x-xgam))
#    return (-1.+2*np.heaviside(x-xgam,1))/np.max(abs(xalph-xgam))

def loc_mat(nodes):                                                            #Computes the element quantities when supplied with gloabl nodes
    xi=GP.xi;W=GP.wght
    x=np.array(nB.Ns(xi)).T @ nodes
    Je=np.array(nB.dN(xi)).T @ nodes
#    print(Je)
    if El[0]=='L' or El[0]=='M':                                               # 1D lagrange or legendre basis
        b1 = np.array(B.Ns(xi)).reshape(-1,1,W.size)
        a1=np.array(B.dN(xi)).reshape(-1,1,W.size)  
        a2=a1.reshape(1,len(a1),-1).copy()
        b2=b1.reshape(1,len(b1),-1).copy()
        a1 *= E(x)/Je*geom.A*W                                                 #multiply by weights, jacobian, A, in bitwise fashion 
        b1 *= geom.C*Je*W
        mat=np.tensordot(a1,a2,axes=([1,2],[0,2])) + np.tensordot(b1,b2,axes=([1,2],[0,2]))     #could potentially use einsum to clean it up (but works for now!)
        elemF=(np.array(B.Ns(xi))*W*Je*T(x)).sum(axis=1).flatten()
        return mat,elemF
    elif El[0]=='G':                                                           #GFEM approximation 
        
        enrich_track_var=nodes.reshape(1,-1).T==enrich_nodes                   #check if any of these element nodes fall under repr./blending elements
        
        halph = nodes[-1] - nodes[0]
        enrch = B.enrich                                                       #enrichment order 
        
        varphi_alph = np.array(B.Ns(xi)).reshape(1,-1,W.size)                  #\varphi_1 evaluated at gauss points (along 3rd Dimensn)
#        print(varphi_alph.shape)
        varphi_alph_p = 1/Je* np.array(B.dN(xi)).reshape(1,-1,W.size)          #\varphi_1' evaluated at gauss points (along 3rd Dimensn)
        
        Ealph = np.array(Ej(x,nodes,halph,enrch))
        Ealphp = np.array(Ejp(x,nodes,halph,enrch))
            
#        varphi_alph_E_alph = np.einsum('ijk,ijk->ijk',varphi_alph,Ealph)
#        print(varphi_alph_E_alph[0,1,:])
#        print(Nf[2,0,:])
        
#        varphi_alph_E_alph_p = np.einsum('ijk,ijk->ijk',varphi_alph_p,Ealph) + np.einsum('ijk,ijk->ijk',varphi_alph,Ealphp)      
        
#        if enrich_track_var.any():
#            print('Enriched nodes found')
#            phi_gam_x_alph=Etilj(nodes,geom.xgam)                              # Value of the enrichment at the nodes (for approximation)
#            phi_gam = (np.einsum('ikj,k->ij',varphi_alph,phi_gam_x_alph)).squeeze()        # PhiGam(x) evaluated at the gauss points 
#            phi_gamp = (np.einsum('ikj,k->ij',varphi_alph,phi_gam_x_alph)).squeeze()       # Gradient of PhiGam(x) evaluated at gauss points
        corr_fac = (varphi_alph[:,find_nodes(nodes),:].sum(axis=1)).squeeze()          # Correction function evaluated at the gauss points 
#            print(corr_fac)
        phi_gam = Etilj(x,geom.xgam)
        phi_gamp = Etiljp(x,geom.xgam)
#        phi_gam *= corr_fac                                                    # corrected
#        phi_gamp *= corr_fac

        phigammod = phi_gam.reshape(1,-1,W.size).repeat(nodes.size,axis=1)
        phigammodP = phi_gamp.reshape(1,-1,W.size).repeat(nodes.size,axis=1)
        Ealph_mod = np.concatenate((Ealph,phigammod))
        Ealphp_mod = np.concatenate((Ealphp,phigammodP))
#        print(Ealphp[0,0,:])
#            plt.plot(x,Ealph_mod[1,1,:])
#        else:
#            Ealph_mod = Ealph.copy()
#            Ealphp_mod = Ealphp.copy()
        
        varphi_alph_E_alph = np.einsum('ijk,ijk->ijk',varphi_alph,Ealph_mod)
        varphi_alph_E_alph_p = np.einsum('ijk,ijk->ijk',varphi_alph_p,Ealph_mod) + np.einsum('ijk,ijk->ijk',varphi_alph,Ealphp_mod)
        
#        print(varphi_alph_E_alph_p)
        Nf = np.einsum('jik',varphi_alph_E_alph).reshape(-1,1,W.size)
        dNf = np.einsum('jik',varphi_alph_E_alph_p).reshape(-1,1,W.size)
        #plt.figure()
        plt.plot(x,phi_gam)
        
        mat = E(x)*geom.A*W*Je*np.einsum('ikm,jkm->ijm',dNf,dNf)
        mat = mat.sum(axis=-1)
#        print(mat.shape)
        M_mat = geom.C*geom.A*W*Je*np.einsum('ikm,jkm->ijm',Nf,Nf) 
        M_mat = M_mat.sum(axis=-1)
        mat += M_mat                                                           # Taking into account any terms with C(x)
        elemF = (W*Je*Nf*T(x)).sum(axis=-1)
        return mat,elemF.flatten()

def fsample(x,dof,nodes,sample_type):                                          #outputs the FE field (i.e. disp here) at x according to approximation
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


Np = 151                                                                        # Order of Gauss-Integration for the Discrete Variational-Problem 
Nel=1
El='G1'                                                                        # 1D Element of degree El[1]
MapX='L1' 
B=basis(float(El[-1]),El[0])                                                   #Basis for FE fields (isoparametric) 
GP=GPXi(Np) 

if El[0]=='M' or El[0]=='L':                                                   #1D Mapping function for the physical coordinates (can take only single digits, some modification needed)
    nB=basis(float(MapX[-1]),MapX[0])
    geom=geometry(Nel,float(MapX[-1])) 
    probsize=geometry(Nel,float(El[-1]))
elif El[0]=='G':
    nB=basis(float(El[-1]),El[0])
    geom=geometry(Nel,float(MapX[-1])) 
    probsize=geometry(Nel,float(El[-1]))

Bp = (geom.E1*geom.E2-g(geom.xgam)*(geom.E2-geom.E1)-g(geom.L)*geom.E1)/(geom.E2*(geom.xgam*(geom.E2-geom.E1)+geom.L*geom.E1))

if El[0]=='M' or El[0]=='L':
    nodes = np.linspace(0.,geom.L,geom.nNNodes)
    elems = np.vstack((np.arange(k,k+nodes.size-1,int(MapX[1])) for k in range(int(MapX[1])+1))).T
    globK=0*np.eye((B.enrich)*probsize.nNNodes)                                     # This changes because now we no longer have iso-p map
    globF=np.zeros((B.enrich)*probsize.nNNodes)    
    prescribed_dof=np.array([[0,0],
                             [-B.enrich,1]])
elif El[0]=='G':
    nodes = np.linspace(0.,probsize.L,probsize.nNNodes)
    elems = np.vstack((np.arange(k,k+nodes.size-1,int(El[1])) for k in range(int(El[1])+1))).T
    enrich_nodesidx=np.arange(0,len(nodes),1)[find_nodes(nodes)]                      # index of the nodes to be enriched 
    enrich_nodes=nodes[find_nodes(nodes)]
    
    globK=0*np.eye((B.enrich+2)*probsize.nNNodes)            # add additional dofs corr. to interface enrichment
    globF=np.zeros((B.enrich+2)*probsize.nNNodes) 
    prescribed_dof=np.array([[0,0]])#,
#                             [-B.enrich-2,1]])


dof=np.inf*np.ones(len(globK))                                                 # initialize to infinity 
prescribed_forc=np.array([[-B.enrich-2,2],
                          [-B.enrich-1,2]])                              # Don't use `int` here! (prescribed concentrated loads) 
dof[prescribed_dof[:,0]]=prescribed_dof[:,1]
#dof[1]=0
for k in range(elems[:,0].size):                                               #Assembly of Global System
    elnodes=elems[k]
    if El[0]=='M' or El[0]=='L':
        elemord=int(El[-1])+1
        globdof = np.array([k*(elemord-1)+i for i in range(B.enrich*elemord)],int)
    elif El[0]=='G':
        elemord = B.enrich+1+int(El[-1])
        ndofsel=(int(El[-1])+1)*(B.enrich+2)
        globdof = np.array([k*(ndofsel-B.enrich-2)+i for i in range(ndofsel)],int)
#        globdof = globdof[[0,2,3,1]]
    if El[0]=='L' or El[0]=='M' or El[0]=='G':                                               # 1D lagrange or legendre polynomials 
        nodexy=nodes[elnodes]
    else:
        nodexy=nodexy[:,elnodes]
    kel,fel=loc_mat(nodexy)
    globK[np.ix_(globdof,globdof)] += kel                                      #this process would change in vectorization 
    globF[globdof] += fel

globF[prescribed_forc[:,0]] += prescribed_forc[:,1]
#print(globF[-1])
fdof=dof==np.inf                                                               # free dofs
nfdof=np.invert(fdof)                                                          # specified dofs 


if El[0]=='L' or El[0]=='M':  
    AA=globK[np.ix_(fdof,fdof)]
    print(nla.cond(globK[np.ix_(fdof,fdof)]))
    bb=globF[fdof]-globK[np.ix_(fdof,nfdof)] @ dof[nfdof]                                                                           
    dofme=nla.solve(AA,bb)
elif El[0]=='G':
    AA=sp.csr_matrix( globK[np.ix_(fdof,fdof)])
    print(nla.cond(globK[np.ix_(fdof,fdof)]))
    bb=globF[fdof]-globK[np.ix_(fdof,nfdof)] @ dof[nfdof]
    dofme,infodof = sla.bicg(AA,bb,tol=1.e-15)                                 #Use Bi-Conjugate Gradient to Solve Ax=b (?)
dof[fdof]=dofme.copy()
    
if El[0] is 'L' or El[0] is 'M':
    plot_it = True 
else:
    plot_it=False
    
if plot_it:
    xplot=np.linspace(nodes[0],nodes[-1],100)
    ue=uex(xplot)
    xpl2 = np.linspace(nodes[0],nodes[-1],10*geom.nQNodes-1)
    fittype='vandermonde'                                                          #how to plot solution: Classical --> Vandermonde (using local appx), Fast --> Splines (default splrep)
    if fittype=='spline':
        xpl2,uapp=fsample(xpl2,dof[0:len(dof):int(El[1])],nodes[0:len(nodes):2],fittype)
    elif fittype=='vandermonde':
        xpl2,uapp=fsample(xpl2,dof,nodes,fittype)
    
    #Solvnig the problem in case of nonzero C (where it gets nasty) numerically using collocation 
    y_a=np.zeros((2,nodes.size))
    #y_a[0,-1]=1
    y_b=np.zeros((2,nodes.size))
    res_a=solve_bvp(fexact,fbc,nodes,y_a)
    
    
    #Plotting the solution (make it separate once you test cases)
    plt.figure(figsize=(8,8))
    plt.plot(xplot,uex(xplot),label=r'Exact Solution')
    #plt.plot(xplot,res_a.sol(xplot)[0],label=r'Scipy: solve\_bvp')
    ax=plt.gca()
    if El[0]=='L':
        plt.plot(xpl2,uapp,'rx',label=r'Quadratic Lagrange FE')
    elif El[0]=='M':
        plt.plot(xpl2,uapp,'rx',label=r'p-Hierarchical GFEM: Degree '+str(El[1]))
    plt.tick_params(axis='both',which='both',direction='in',top=True,right=True,labelsize=18)
    plt.title(r'Number of elements: '+str(Nel)+', NGP = '+str(Np),fontsize=20)
    plt.legend(loc=0,fontsize=18)
    plt.grid(True,linestyle='--')
    plt.text(0.1,0.4,r'$U^h$ = '+str(0.5*dof @ (globK @ dof)),transform=ax.transAxes,
             fontsize=22,bbox=dict(facecolor='red', alpha=0.5))
print('Uh=',0.5*dof @ (globK @ dof))
