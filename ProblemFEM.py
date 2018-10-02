# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:37:11 2018

@author: bshrima2

Sparse Matrix as such not needed for the small dimensionality of the problem!
Solving a 1D BVP:
    -(EA(x)u'(x))'+Cu=T  for x in (0,L)
    u(0)=u(1)=0  : Homogeneous BC's

with quadratic Lagrange elements 
Parameters: 
    
    L  = Length of the domain
    A  = Area function A(x)




FIX ME: 
    change the exact solution for plotting 
    include p-hierarchical shape functions (FEM not GFEM)
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.integrate import quad,solve_bvp
import scipy.sparse.linalg as sla
import scipy.sparse as sp
from scipy.interpolate import splrep,splev

plt.rc('text',usetex=True)

class geometry() : #1D geometry parameters    
    def __init__(self,ne,deg):
        self.L=1.
        self.A=1.
        self.E=1.
        self.C=0.
        self.a=50.
        self.xb=0.2
        self.nLNodes=ne+1       #Linear elements
        self.nQNodes=2*ne+1     #Quadratic elements 
        self.nNNodes=ne*deg+1   #Isoparametric deg-th order lagrange elements 
        self.ndim=1
        self.M=0
        self.g=0.

def uex(x,a,xb):
    return (1-x)*(np.arctan(a*(x-xb))+np.arctan(a*xb))

def T(x,a,xb):
    axb = a*(x-xb)
    return 2.*a/(1.+axb**2)+2*(1-x)*a**3/(1+axb**2)**2*(x-xb)

class GPXi():
    def __init__(self,ordr):
        from numpy.polynomial.legendre import leggauss   #Gauss-Legendre Quadrature for 1D
#        from numpy.polynomial.chebyshev import chebgauss #Gauss-Chebyshev Quadrature for 1D (bad!)
        self.xi=leggauss(ordr)[0]
        self.wght=leggauss(ordr)[1]

class basis():  # defined on the canonical element (1D : [-1,1] ) 
    def __init__(self,deg,basis_type):
        from sympy import Symbol,diff,Array,lambdify,legendre,integrate
        if basis_type == 'L':                                                  #1D Lagrange 
            z=Symbol('z')
            if deg==2.:     # denotes the number of nodes 
                N=1/2*Array([1-z,1+z])
                dfN=diff(N,z)
                self.Ns=lambdify(z,N,'numpy')
                self.dN=lambdify(z,dfN,'numpy')
            elif deg==3.:
                N=1/2*Array([z*(z-1),2*(1+z)*(1-z),z*(1+z)])
                dfN=diff(N,z)
                self.Ns=lambdify(z,N,'numpy')
                self.dN=lambdify(z,dfN,'numpy')
            elif deg==4.:                                                      #not implemented yet. the no. of nodes change. define a general geom.nNnodes (?)
                N=1/2*Array([z*(z-1),2*(1+z)*(1-z),z*(1+z)])
                dfN=diff(N,z)
                self.Ns=lambdify(z,N,'numpy')
                self.dN=lambdify(z,dfN,'numpy')
            else:
                raise Exception('Element type not implemented yet')
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
            dfN=diff(N,z)+1.e-25*N
            self.Ns=lambdify(z,N,'numpy')
            self.dN=lambdify(z,dfN,'numpy')
            
def fexact(x,y,p):
    u,up=y
    a,xb=p
    return [up,1/(geom.E*geom.A)*(geom.C*u-T(x,a,xb))]

def fbc(ya,yb,p):
    a,xb=p
    return np.array([ya[0],yb[0],a-geom.a,xb-geom.xb])


El='L3'               #Linear 3-noded element (--> Quadratic Lagrange)
B=basis(float(El[1]),El[0]) #Basis for FE fields (isoparametric) 
Np = 20              #Order of Gauss-Integration for the Discrete Galerkin-Form 
Nel=10
geom=geometry(Nel,float(El[1]))
GP=GPXi(Np)

def loc_mat(nodes,a,xb):           #Computes the element quantities when supplied with gloabl nodes
    xi=GP.xi;W=GP.wght
    if El[1]=='2': # fix this if condition 
        x=np.array(B.Ns(xi)).T @ nodes    # Mapping for linear 1D
        Je=(np.array(B.dN(xi)).T @ nodes)*np.ones(xi.size,)        #Jacobian for 1D mapping
    else:
        x=x=np.array(B.Ns(xi)).T @ nodes
        Je=np.array(B.dN(xi)).T @ nodes
    if El=='L2' or El=='L3':   # 1D elements
        b1 = np.array(B.Ns(xi)).reshape(-1,1,W.size)
        if El=='L2':
            a1 = np.array(B.dN(xi)).repeat(W.size).reshape(2,-1).reshape(-1,1,W.size)
        else: 
            a1=np.array(B.dN(xi)).reshape(-1,1,W.size)  
        a2=a1.reshape(1,len(a1),-1).copy()
        b2=b1.reshape(1,len(b1),-1).copy()
        a1 *= geom.E/Je*geom.A*W                 #multiply by weights, jacobian, A, bitwise 
        b1 *= geom.C*Je*W
        mat=np.tensordot(a1,a2,axes=([1,2],[0,2])) + np.tensordot(b1,b2,axes=([1,2],[0,2]))
#        elemF=np.dstack((np.array(B.Ns(xi[i]))*W[i]*Je[i]*b(x[i]) for i in range(xi.size)))
#        elemF=elemF.sum(axis=2).flatten()
        elemF=(np.array(B.Ns(xi))*W*Je*T(x,a,xb)).sum(axis=1).flatten()
        return mat,elemF


if float(El[1])==2.:
    nodes=np.linspace(0.,geom.L,geom.nLNodes) # Global nodal coordinates 
    elems=np.vstack( (np.arange(1,geom.nLNodes,1),np.arange(2,geom.nLNodes+1,1)) ).astype(int).T-1 # only for linear 1D lagrange 
elif float(El[1])==3.:
    nodes=np.linspace(0.,geom.L,geom.nQNodes) # Global nodal coordinates
    elems=np.vstack( (np.arange(1,geom.nQNodes,2),
                      np.arange(2,geom.nQNodes+1,2),
                      np.arange(3,geom.nQNodes+2,2))).astype(int).T-1 # only for quadr. 1D lagrange

globK=0*np.eye(len(nodes)*geom.ndim)
globF=np.zeros(len(nodes)*geom.ndim)
dof=np.inf*np.ones(len(globK))     #initialize 
prescribed_dof=np.array([[0,0],
                         [-1,0]])
prescribed_forc=0*np.array([[-1,-geom.M*geom.g]]) # Don't use `int` here! (prescribed concentrated loads) 

dof[prescribed_dof[:,0]]=prescribed_dof[:,1]

for k in range(elems[:,0].size):  #assembly
    elnodes=elems[k]
    if El[1]=='2':
        globdof=np.array([geom.ndim*elnodes[0],geom.ndim*elnodes[1]])
    elif El[1]=='3':
        globdof=np.array([geom.ndim*elnodes[0],geom.ndim*elnodes[1],geom.ndim*elnodes[2]])
    if El[0]=='L':
        nodexy=nodes[elnodes]
    else:
        nodexy=nodexy[:,elnodes]
    kel,fel=loc_mat(nodexy,geom.a,geom.xb)
    globK[np.ix_(globdof,globdof)] += kel  #this process would change in vectorization 
    globF[globdof] += fel

globF[int(prescribed_forc[:,0])] += prescribed_forc[:,1]
#print(globF[-1])
fdof=dof==np.inf            # free dofs
nfdof=np.invert(fdof)       # specified dofs 
dof[fdof]=la.solve(globK[np.ix_(fdof,fdof)],globF[fdof]-globK[np.ix_(fdof,nfdof)] @ dof[nfdof])
xplot=np.linspace(0.,geom.L,1000)  #sampling points for the exact solution




def plotfigs(xp):
    plt.figure(1)
    plt.plot(xp,fexact(xp)['disp'],label=r'Exact Solution')
    if El[1]=='2':
        plt.plot(xp,fsample(xp,dof)['disp'],label=r'Linear Basis')
    elif El[1]=='3':
        plt.plot(xp,fsample(xp,dof)['disp'],label=r'Quadratic Basis')
    plt.legend(loc=0,fontsize=14)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    ax=plt.gca()
#    ax.xaxis.set_minor_locator(mx)
    ax.set_title(r'Displacement',fontsize=14)
#    ax.yaxis.set_minor_locator(my)    



    plt.figure(2)
    plt.plot(xp,fexact(xp)['strain'],label=r'Exact Solution')
    if El[1]=='2':
        plt.plot(xp,fsample(xp,dof)['strain'],label=r'Linear Basis')
    elif El[1]=='3':
        plt.plot(xp,fsample(xp,dof)['strain'],label=r'Quadratic Basis')
    plt.legend(loc=0,fontsize=14)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    ax=plt.gca()
#    ax.xaxis.set_minor_locator(mx)
    ax.set_title(r'Strain',fontsize=14)
#    ax.yaxis.set_minor_locator(my)    



    plt.figure(3)
    plt.plot(xp,fexact(xp)['stress'],label=r'Exact Solution')
    if El[1]=='2':
        plt.plot(xp,fsample(xp,dof)['stress'],label=r'Linear Basis')
    elif El[1]=='3':
        plt.plot(xp,fsample(xp,dof)['stress'],label=r'Quadratic Basis')
    plt.legend(loc=0,fontsize=14)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    ax=plt.gca()
#    ax.xaxis.set_minor_locator(mx)
    ax.set_title(r'Stress',fontsize=14)
#    ax.yaxis.set_minor_locator(my)    

def fsample(x,dof):  #giving out the displacement, strain and stress at x: GP and dof value of disp. at gauss point
    dspl=splrep(nodes,dof,k=int(El[1])-1)
    return splev(x,dspl)

#plotfigs(xplot) 
ue=uex(xplot,geom.a,geom.xb)
xpl2 = np.linspace(0,nodes[-1],10*geom.nQNodes-1)
uapp=fsample(xpl2,dof)
y_a=np.zeros((2,nodes.size))
y_b=np.zeros((2,nodes.size))

res_a=solve_bvp(fexact,fbc,nodes,y_a,p=[geom.a,geom.xb])

plt.figure()
plt.plot(xplot,res_a.sol(xplot)[0],label=r'Exact Solution')
plt.text(0.5,0.6,r'$a=$'+str(geom.a),fontsize=22)
plt.plot(xpl2,uapp,'rx',label=r'Quadratic Lagrange FE')
plt.tick_params(axis='both',labelsize=18)
plt.title(r'Number of elements: '+str(Nel)+', NGP = '+str(Np),fontsize=20)
plt.legend(loc=0,fontsize=18)