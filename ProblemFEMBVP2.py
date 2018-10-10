# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:37:11 2018

@author: bshrima2

Sparse Matrix as such not needed for the small dimensionality of the problem!
Solving a 1D BVP:
    
    -(EA(x)u'(x))'+Cu=T  for x in (0,L)
    u(0)=0; u(L)=1  : Homogeneous BC's

with quadratic Lagrange elements , 
now (hopefully) with quadratic (at least) p-Hierarchical elements, 
and, now (hopefully) with enrichments to PoUs (hat functions)
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
import numpy.linalg as la
from scipy.integrate import quad,solve_bvp
import scipy.sparse.linalg as sla
import scipy.sparse as sp
from scipy.interpolate import splrep,splev,interp1d

plt.rc('text',usetex=True)

class geometry() : #1D geometry parameters    
    def __init__(self,ne,deg):
        self.L=10.
        self.A=1.
        self.E1=10**4
        self.E2=10**3
        self.C=0.
        self.a=50
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
        from sympy import Symbol,diff,Array,lambdify,legendre,integrate,simplify
        if basis_type == 'L':                                                  #1D Lagrange basis of degree deg
            z=Symbol('z')
            Xi=np.linspace(-1,1,deg+1)
            def lag_basis(k):                                                  #may use nprod from mpmath but not worth it, at least in 1d
                n = 1.
                for i in range(len(Xi)):
                    if k != i:
                        n *= (z-Xi[i])/(Xi[k]-Xi[i])
                return n
            N = Array([simplify(lag_basis(m)) for m in range(deg+1)])            
            dfN = diff(N,z)+1.e-25*N
            self.Ns=lambdify(z,N,'numpy')
            self.dN=lambdify(z,dfN,'numpy')
        elif basis_type == 'M':                                                #1D Legendre Polynomials (Fischer calls this spectral)
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
        elif basis_type == 'GFEM1D1':
            z=Symbol('z')
            x=Symbol('x')
            Xi=np.linspace(-1,1,deg+1)
            def lag_basis(k):                                                  #may use nprod from mpmath but not worth it, at least in 1d
                n = 1.
                for i in range(len(Xi)):
                    if k != i:
                        n *= (z-Xi[i])/(Xi[k]-Xi[i])
                return n
            N1 = Array([simplify(lag_basis(m)) for m in range(deg+1)])            
            dfN = diff(N1,z)+1.e-25*N1
#            Just enrich the nodes whose support xgam lies in??
#            
#        else:
#            raise Exception('Element type not implemented yet')
       
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
    return 25*x-7.5*x**2+0.5*x**3

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

El='M6'                                                                        #1D Element of degree El[1]
MapX='L3'                                                                      #1D Mapping function for the physical coordinates (can take only single digits, some modification needed)
nB=basis(float(MapX[-1]),MapX[0])
B=basis(float(El[-1]),El[0])                                                   #Basis for FE fields (isoparametric) 
Np = 10                                                                         #Order of Gauss-Integration for the Discrete Galerkin-Form 
Nel=5
geom=geometry(Nel,float(MapX[-1])) 
probsize=geometry(Nel,float(El[-1]))
GP=GPXi(Np)
Bp = (geom.E1*geom.E2-g(geom.xgam)*(geom.E2-geom.E1)-g(geom.L)*geom.E1)/(geom.E2*(geom.xgam*(geom.E2-geom.E1)+geom.L*geom.E1))

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

nodes = np.linspace(0.,geom.L,geom.nNNodes)
elems = np.vstack((np.arange(k,k+nodes.size-1,int(MapX[1])) for k in range(int(MapX[1])+1))).T


globK=0*np.eye(probsize.nNNodes)                                               # This changes because now we no longer have iso-p map
globF=np.zeros(probsize.nNNodes)                                               # This changes too, again because no iso-p map 
dof=np.inf*np.ones(len(globK))                                                 #initialize 
prescribed_dof=np.array([[0,0],
                         [-1,1]])
prescribed_forc=0*np.array([[-1,-geom.M*geom.g]])                              # Don't use `int` here! (prescribed concentrated loads) 

dof[prescribed_dof[:,0]]=prescribed_dof[:,1]

for k in range(elems[:,0].size):                                               #Assembly of Global System
    elnodes=elems[k]
    if El[0]=='M' or El[0]=='L':
        elemord=int(El[-1])+1
        globdof = np.array([k*(elemord-1)+i for i in range(elemord)],int)
    if El[0]=='L' or El[0]=='M':                                               # 1D lagrange or legendre polynomials 
        nodexy=nodes[elnodes]
    else:
        nodexy=nodexy[:,elnodes]
    kel,fel=loc_mat(nodexy)
    globK[np.ix_(globdof,globdof)] += kel                                      #this process would change in vectorization 
    globF[globdof] += fel

globF[int(prescribed_forc[:,0])] += prescribed_forc[:,1]
#print(globF[-1])
fdof=dof==np.inf                                                               # free dofs
nfdof=np.invert(fdof)                                                          # specified dofs 
dof[fdof]=la.solve(globK[np.ix_(fdof,fdof)],globF[fdof]-
   globK[np.ix_(fdof,nfdof)] @ dof[nfdof])
xplot=np.linspace(0.,geom.L,1000)                                              #sampling points for the exact solution

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
print(0.5*dof @ (globK @ dof))
