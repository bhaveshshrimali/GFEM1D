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
    Check the order in which the shape functions are being returned
    
    
    
    
    
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as nla
from scipy.integrate import quad,solve_bvp
import scipy.sparse.linalg as sla
import scipy.sparse as sp
from scipy.interpolate import splrep,splev,interp1d
from sympy import Symbol, Array, diff, integrate, simplify, lambdify, legendre, Abs, tensorproduct, flatten, transpose
from copy import deepcopy 
plt.rc('text',usetex=True)


class geometry() : #1D geometry parameters    
    def __init__(self,ne,deg):
        self.L=10
        self.A=1.
        self.E1=10**4
#        self.E2=2*self.E1
        self.E2=10**3
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
#        print(basis_type)
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
        elif basis_type == 'M':                                                # 1D Legendre Polynomials 
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
            self.enrich=enrch                                                  #Enrichment order corresponding to GFEM shape functions 
            xalph = Symbol('xalph',real=True)
            halph = Symbol('halph',real=True)
            N1 = Array([((z-xalph)/halph)**n + 1.e-26*(z-xalph)/halph for n in range(self.enrich+1)])   #shape consistency take E_o = 1
            Nf1 = lambdify((z,xalph,halph),N1,'numpy')
            dfN1 = lambdify((z,xalph,halph),diff(N1,z)+1.e-26*N1,'numpy')
            self.Np=Nf1
            self.dNp=dfN1
        elif basis_type == 'IF':
#            print('here')
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
            self.enrich=enrch                                                  # Enrichment order corresponding to GFEM shape functions 
            xalph = Symbol('xalph',real=True)
            halph = Symbol('halph',real=True)
            xgam=Symbol('xgam',real=True)
            
            Nf2 = simplify( Array([(abs(z-xgam))**m * ((z-xalph)/halph)**n +1.e-26*(z-xalph)/halph for m in range(2) for n in range(enrch+1)]) )  # Enrichment by non-polynomials (for interfaces) 
            dfN2 = simplify(Nf2.diff(z) + 1.e-26*Nf2)
            self.NIF=lambdify((z,xgam,xalph,halph),Nf2,'numpy')                # Final enrichment functions including both polynomial and level set
            self.DNIF=lambdify((z,xgam,xalph,halph),dfN2,'numpy')
        
        elif basis_type == 'IFM':
#            print('here')
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
            self.enrich=enrch                                                  # Enrichment order corresponding to GFEM shape functions 
            xalph = Symbol('xalph',real=True)
            halph = Symbol('halph',real=True)
            xgam=Symbol('xgam',real=True)
            
            Nf2 = simplify( Array([(abs(z-xalph))**m * ((z-xalph)/halph)**n +1.e-26*(z-xalph)/halph for m in range(2) for n in range(enrch+1)]) )  # Enrichment by non-polynomials (for interfaces) 
            dfN2 = simplify(Nf2.diff(z) + 1.e-26*Nf2)
            
            Nf3 = simplify( Array([((z-xalph)/halph)**n +1.e-26*(z-xalph)/halph for n in range(enrch+1)]) )  # Enrichment by non-polynomials (for interfaces) 
            dfN3 = simplify(Nf3.diff(z) + 1.e-26*Nf3)
            
            self.NIF=lambdify((z,xgam,xalph,halph),Nf3,'numpy')                # Final enrichment functions including both polynomial and level set
            self.DNIF=lambdify((z,xgam,xalph,halph),dfN3,'numpy')
        
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
    return abs(nodes-geom.xgam) < geom.L/Nel                                   # Equal to will create some problems     

def Etilj(x,xgam):                                                             # Absolute signed distance function with smoothening 
    
    return abs(x-xgam) 

def Etiljp(x,xgam):
    return np.heaviside(x-xgam,0.)/(geom.L-xgam)
#    return np.sign(x-xgam)*np.sign((x-xgam)*np.sign(x-xgam))
#    return (-1.+2*np.heaviside(x-xgam,1))/np.max(abs(xalph-xgam))

def loc_mat(nodes,ElFlag,Ndnode):                                                     #Computes the element quantities when supplied with gloabl nodes
    xi=GP.xi;W=GP.wght
    
    if etypes=='M' or etypes=='L' or etypes=='G':
        x=np.array(nB.Ns(xi)).T @ nodes
        Je=np.array(nB.dN(xi)).T @ nodes
    elif etypes=='IF' or etypes == 'IFM':
        x=np.array(B.Ns(xi)).T @ nodes
        Je=np.array(B.dN(xi)).T @ nodes

#    Nodes and weights for integration subcells    
#        print(nodes>geom.xgam)
#        print(np.hstack((geom.xgam,nodes[nodes>geom.xgam])).shape)
        
#    print(Je)
    if etypes=='L' or etypes=='M':                                               # 1D lagrange or legendre basis
        b1 = np.array(B.Ns(xi)).reshape(-1,1,W.size)
        a1=np.array(B.dN(xi)).reshape(-1,1,W.size)  
        a2=a1.reshape(1,len(a1),-1).copy()
        b2=b1.reshape(1,len(b1),-1).copy()
        a1 *= E(x)/Je*geom.A*W                                                 # multiply by weights, jacobian, A, in bitwise fashion 
        b1 *= geom.C*Je*W
        mat=np.tensordot(a1,a2,axes=([1,2],[0,2])) + np.tensordot(b1,b2,axes=([1,2],[0,2]))     #could potentially use einsum to clean it up (but works for now!)
        elemF=(np.array(B.Ns(xi))*W*Je*T(x)).sum(axis=1).flatten()
        return mat,elemF
    elif etypes=='G':                                                           # GFEM approximation (with only polynomial enrichments)

        halph = nodes[-1] - nodes[0]        
        varphi_alph = np.array(B.Ns(xi)).reshape(1,-1,W.size)                  # \varphi_1 evaluated at gauss points (along 3rd Dimensn)
#        print(varphi_alph.shape)
        varphi_alph_p = 1/Je* np.array(B.dN(xi)).reshape(1,-1,W.size)          # \varphi_1' evaluated at gauss points (along 3rd Dimensn)
        
        Ealph= np.array(B.Np(x,nodes.reshape(-1,1),halph))                     # broadcast the nodes to vectorize calculations on the fly
        Ealphp=np.array(B.dNp(x,nodes.reshape(-1,1),halph))
#        print(Ealph.shape)
        
        varphi_alph_E_alph = np.einsum('ijk,ijk->ijk',varphi_alph,Ealph)
        varphi_alph_E_alph_p = np.einsum('ijk,ijk->ijk',varphi_alph_p,Ealph) + np.einsum('ijk,ijk->ijk',varphi_alph,Ealphp)
        
#        print(varphi_alph_E_alph_p)
        Nf = np.einsum('jik',varphi_alph_E_alph).reshape(-1,1,W.size)
        dNf = np.einsum('jik',varphi_alph_E_alph_p).reshape(-1,1,W.size)

        mat = E(x)*geom.A*W*Je*np.einsum('ikm,jkm->ijm',dNf,dNf)
        mat = mat.sum(axis=-1)
#        print(mat.shape)
        M_mat = geom.C*geom.A*W*Je*np.einsum('ikm,jkm->ijm',Nf,Nf) 
        M_mat = M_mat.sum(axis=-1)
        mat += M_mat                                                           # Taking into account any terms with C(x)
        elemF = (W*Je*Nf*T(x)).sum(axis=-1)
        return mat,elemF.flatten()
    elif etypes == 'IF':
        halph = nodes[-1]-nodes[0]
#        Find out if the current element is reproducing element (to split integration domain)
        if ElFlag != 'rpp':
            varphi_alph = np.array(B.Ns(xi)).reshape(1,-1,W.size)
            varphi_alph_p = 1/Je* np.array(B.dN(xi)).reshape(1,-1,W.size)          # \varphi_1' evaluated at gauss points (along 3rd Dimensn)
        
            Ealph= np.array(B.NIF(x,geom.xgam,nodes.reshape(-1,1),halph))          # broadcast the nodes to vectorize calculations on the fly
            Ealphp=np.array(B.DNIF(x,geom.xgam,nodes.reshape(-1,1),halph))
            #        print(Ealph.shape)
        
            varphi_alph_E_alph = np.einsum('ijk,ijk->ijk',varphi_alph,Ealph)
            varphi_alph_E_alph_p = np.einsum('ijk,ijk->ijk',varphi_alph_p,Ealph) + np.einsum('ijk,ijk->ijk',varphi_alph,Ealphp)
        
            Nf = np.einsum('jik',varphi_alph_E_alph).reshape(-1,1,W.size)
            dNf = np.einsum('jik',varphi_alph_E_alph_p).reshape(-1,1,W.size)

            mat = E(x)*geom.A*W*Je*np.einsum('ikm,jkm->ijm',dNf,dNf)
            mat = mat.sum(axis=-1)
#        print(mat.shape)
            M_mat = geom.C*geom.A*W*Je*np.einsum('ikm,jkm->ijm',Nf,Nf) 
            M_mat = M_mat.sum(axis=-1)
            mat += M_mat                                                           # Taking into account any terms with C(x)
            elemF = (W*Je*Nf*T(x)).sum(axis=-1)
            #        Now return only the relevant DOFS for mat and elemF

        elif ElFlag=='rpp':
            
            nodes1 = np.hstack((nodes[nodes<geom.xgam],geom.xgam))                 #node1, node2,...,xgam
            nodes2 = np.hstack((geom.xgam,nodes[nodes>geom.xgam]))                 #xgam,........,nodeN
            
            x1 = np.array(B.Ns(xi)).T @ nodes1
            x2 = np.array(B.Ns(xi)).T @ nodes2
            
            xi_1 = (2*x1.copy()-nodes.sum())/halph
            xi_2 = (2*x2.copy()-nodes.sum())/halph
            
#            print(xi_2)
            
            
            x11 = np.array(B.Ns(xi_1)).T @ nodes
            x21 = np.array(B.Ns(xi_2)).T @ nodes
            
#            print(nodes.shape)
#            print(x2.size)
            
            Je1 = np.array(B.dN(xi)).T @ nodes1
            Je2 = np.array(B.dN(xi)).T @ nodes2 
            
#            print(Je2)
                      

            varphi_alph1 = np.array(B.Ns(xi_1)).reshape(1,-1,W.size)
            varphi_alph2 = np.array(B.Ns(xi_2)).reshape(1,-1,W.size)
            
            varphi_alph_p1 = 1/Je* np.array(B.dN(xi_1)).reshape(1,-1,W.size)          # \varphi_1' evaluated at gauss points (along 3rd Dimensn)
            varphi_alph_p2 = 1/Je* np.array(B.dN(xi_2)).reshape(1,-1,W.size)
            
            Ealph1  = np.array(B.NIF(x11,geom.xgam,nodes.reshape(-1,1),halph))          # broadcast the nodes to vectorize calculations on the fly
            Ealphp1 = np.array(B.DNIF(x11,geom.xgam,nodes.reshape(-1,1),halph))
            
            Ealph2= np.array(B.NIF(x21,geom.xgam,nodes.reshape(-1,1),halph))          # broadcast the nodes to vectorize calculations on the fly
            Ealphp2=np.array(B.DNIF(x21,geom.xgam,nodes.reshape(-1,1),halph))

        
            varphi_alph_E_alph1 = np.einsum('ijk,ijk->ijk',varphi_alph1,Ealph1)
            varphi_alph_E_alph_p1 = np.einsum('ijk,ijk->ijk',varphi_alph_p1,Ealph1) + np.einsum('ijk,ijk->ijk',varphi_alph1,Ealphp1)
            
            varphi_alph_E_alph2 = np.einsum('ijk,ijk->ijk',varphi_alph2,Ealph2)
            varphi_alph_E_alph_p2 = np.einsum('ijk,ijk->ijk',varphi_alph_p2,Ealph2) + np.einsum('ijk,ijk->ijk',varphi_alph2,Ealphp2)
            
            
            Nf1 = np.einsum('jik',varphi_alph_E_alph1).reshape(-1,1,W.size)
            dNf1 = np.einsum('jik',varphi_alph_E_alph_p1).reshape(-1,1,W.size)
            
            Nf2 = np.einsum('jik',varphi_alph_E_alph2).reshape(-1,1,W.size)
            dNf2 = np.einsum('jik',varphi_alph_E_alph_p2).reshape(-1,1,W.size)

#            print('dN1 = ',dNf1[:,:,-1])
#            print('dN2 = ',dNf2[:,:,0])

            mat1 = E(x1)*geom.A*W*Je1*np.einsum('ikm,jkm->ijm',dNf1,dNf1)
            mat2 = E(x2)*geom.A*W*Je2*np.einsum('ikm,jkm->ijm',dNf2,dNf2)
            
#            mat1 = mat.sum(axis=-1)
#        print(mat.shape)
            M_mat1 = geom.C*geom.A*W*Je1*np.einsum('ikm,jkm->ijm',Nf1,Nf1) 
            M_mat2 = geom.C*geom.A*W*Je2*np.einsum('ikm,jkm->ijm',Nf2,Nf2)
#            M_mat = M_mat.sum(axis=-1)
            mat = (M_mat1 + M_mat2 + mat1 + mat2).sum(axis=-1)                   # Taking into account any terms with C(x)
            
            elemF = ( W*Je1*Nf1*T(x1) + W*Je2*Nf2*T(x2) ).sum(axis=-1)

            
        if ElFlag == 'nn':
            arrdof=[i for i in range(Ndnode)]+[2*Ndnode+i for i in range(Ndnode)] #which dofs are selected 
        elif ElFlag == 'blr':
            arrdof=[i for i in range(3*Ndnode)]
        elif ElFlag == 'bll':
            arrdof=[i for i in range(Ndnode)]+[2*Ndnode+i for i in range(2*Ndnode)] 
        elif ElFlag == 'rpp':
            arrdof=[i for i in range(len(mat))]
        
        dofsend=np.array(arrdof)
#        print(mat.shape)
        return mat[np.ix_(dofsend,dofsend)],elemF[dofsend].flatten()
    
    elif etypes == 'IFM':
        halph = nodes[-1]-nodes[0]
#        Find out if the current element is reproducing element (to split integration domain)
        if ElFlag != 'rpp':
            varphi_alph = np.array(B.Ns(xi)).reshape(1,-1,W.size)
            varphi_alph_p = 1/Je* np.array(B.dN(xi)).reshape(1,-1,W.size)          # \varphi_1' evaluated at gauss points (along 3rd Dimensn)
        
            Ealph= np.array(B.NIF(x,geom.xgam,nodes.reshape(-1,1),halph))          # broadcast the nodes to vectorize calculations on the fly
            Ealphp=np.array(B.DNIF(x,geom.xgam,nodes.reshape(-1,1),halph))
            #        print(Ealph.shape)
            
            varphi_alph_E_alph = np.einsum('ijk,ijk->ijk',varphi_alph,Ealph)
            varphi_alph_E_alph_p = np.einsum('ijk,ijk->ijk',varphi_alph_p,Ealph) + np.einsum('ijk,ijk->ijk',varphi_alph,Ealphp)
        
            Nf = np.einsum('jik',varphi_alph_E_alph).reshape(-1,1,W.size)
            dNf = np.einsum('jik',varphi_alph_E_alph_p).reshape(-1,1,W.size)

            mat = E(x)*geom.A*W*Je*np.einsum('ikm,jkm->ijm',dNf,dNf)
            mat = mat.sum(axis=-1)
#        print(mat.shape)
            M_mat = geom.C*geom.A*W*Je*np.einsum('ikm,jkm->ijm',Nf,Nf) 
            M_mat = M_mat.sum(axis=-1)
            mat += M_mat                                                           # Taking into account any terms with C(x)
            elemF = (W*Je*Nf*T(x)).sum(axis=-1)
            #        Now return only the relevant DOFS for mat and elemF

        elif ElFlag=='rpp':
            
            nodes1 = np.hstack((nodes[nodes<geom.xgam],geom.xgam))                 #node1, node2,...,xgam
            nodes2 = np.hstack((geom.xgam,nodes[nodes>geom.xgam]))                 #xgam,........,nodeN
            
            x1 = np.array(B.Ns(xi)).T @ nodes1
            x2 = np.array(B.Ns(xi)).T @ nodes2
            
            xi_1 = (2*x1.copy()-nodes.sum())/halph
            xi_2 = (2*x2.copy()-nodes.sum())/halph
            
#            print(xi_2)
            
            
            x11 = np.array(B.Ns(xi_1)).T @ nodes
            x21 = np.array(B.Ns(xi_2)).T @ nodes
            
#            print(nodes.shape)
#            print(x2.size)
            
            Je1 = np.array(B.dN(xi)).T @ nodes1
            Je2 = np.array(B.dN(xi)).T @ nodes2 
            
#            print(Je2)
                      

            varphi_alph1 = np.array(B.Ns(xi_1)).reshape(1,-1,W.size)
            varphi_alph2 = np.array(B.Ns(xi_2)).reshape(1,-1,W.size)
            
            varphi_alph_p1 = 1/Je* np.array(B.dN(xi_1)).reshape(1,-1,W.size)          # \varphi_1' evaluated at gauss points (along 3rd Dimensn)
            varphi_alph_p2 = 1/Je* np.array(B.dN(xi_2)).reshape(1,-1,W.size)
            
            Ealph1  = np.array(B.NIF(x11,geom.xgam,nodes.reshape(-1,1),halph))          # broadcast the nodes to vectorize calculations on the fly
            Ealphp1 = np.array(B.DNIF(x11,geom.xgam,nodes.reshape(-1,1),halph))
            
            Ealph2= np.array(B.NIF(x21,geom.xgam,nodes.reshape(-1,1),halph))          # broadcast the nodes to vectorize calculations on the fly
            Ealphp2=np.array(B.DNIF(x21,geom.xgam,nodes.reshape(-1,1),halph))

        
            varphi_alph_E_alph1 = np.einsum('ijk,ijk->ijk',varphi_alph1,Ealph1)
            varphi_alph_E_alph_p1 = np.einsum('ijk,ijk->ijk',varphi_alph_p1,Ealph1) + np.einsum('ijk,ijk->ijk',varphi_alph1,Ealphp1)
            
            varphi_alph_E_alph2 = np.einsum('ijk,ijk->ijk',varphi_alph2,Ealph2)
            varphi_alph_E_alph_p2 = np.einsum('ijk,ijk->ijk',varphi_alph_p2,Ealph2) + np.einsum('ijk,ijk->ijk',varphi_alph2,Ealphp2)
            
            
            Nf1 = np.einsum('jik',varphi_alph_E_alph1).reshape(-1,1,W.size)
            dNf1 = np.einsum('jik',varphi_alph_E_alph_p1).reshape(-1,1,W.size)
            
            Nf2 = np.einsum('jik',varphi_alph_E_alph2).reshape(-1,1,W.size)
            dNf2 = np.einsum('jik',varphi_alph_E_alph_p2).reshape(-1,1,W.size)

#            print('dN1 = ',dNf1[:,:,-1])
#            print('dN2 = ',dNf2[:,:,0])

            mat1 = E(x1)*geom.A*W*Je1*np.einsum('ikm,jkm->ijm',dNf1,dNf1)
            mat2 = E(x2)*geom.A*W*Je2*np.einsum('ikm,jkm->ijm',dNf2,dNf2)
            
#            mat1 = mat.sum(axis=-1)
#        print(mat.shape)
            M_mat1 = geom.C*geom.A*W*Je1*np.einsum('ikm,jkm->ijm',Nf1,Nf1) 
            M_mat2 = geom.C*geom.A*W*Je2*np.einsum('ikm,jkm->ijm',Nf2,Nf2)
#            M_mat = M_mat.sum(axis=-1)
            mat = (M_mat1 + M_mat2 + mat1 + mat2).sum(axis=-1)                   # Taking into account any terms with C(x)
            
            elemF = ( W*Je1*Nf1*T(x1) + W*Je2*Nf2*T(x2) ).sum(axis=-1)

            
        if ElFlag == 'nn':
            arrdof=[i for i in range(Ndnode)]+[2*Ndnode+i for i in range(Ndnode)] #which dofs are selected 
        elif ElFlag == 'blr':
            arrdof=[i for i in range(3*Ndnode)]
        elif ElFlag == 'bll':
            arrdof=[i for i in range(Ndnode)]+[2*Ndnode+i for i in range(2*Ndnode)] 
        elif ElFlag == 'rpp':
            arrdof=[i for i in range(len(mat))]
        
        dofsend=np.array(arrdof)
#        print(mat.shape)
        return mat[np.ix_(dofsend,dofsend)],elemF[dofsend].flatten()

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

        dofs=dof[dofarang]
        nodesx=nodes[xarang]
        soln = np.einsum('ikj,kmj->imj',N,dofs)
        xpl = np.einsum('ikj,kmj->imj',Nx,nodesx)
        return (xpl.T.ravel(),soln.T.ravel())                                  # return the solution evaluated at the given points 

def babuskasolve(K,f):
    T=np.diag(np.diag(K)**(-0.5))
    Keps=T @ K @ T 
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
    return T @ u
    


enrch=6
Np = 52                                                                       # Order of Gauss-Integration for the Discrete Variational-Problem 
Nel=5
mid_elem_idx=int(Nel/2)
El='IF1'                                                                        # 1D Element of degree El[1]
MapX='L1' 
etypes="".join(El.rsplit(El[-1]))
B=basis(float(El[-1]),etypes)                                                   # Basis for FE fields (isoparametric) 
GP=GPXi(Np) 

if etypes =='M' or etypes =='L':                                                # 1D Mapping function for the physical coordinates (can take only single digits, some modification needed)
    nB=basis(float(MapX[-1]),MapX[0])
    geom=geometry(Nel,float(MapX[-1])) 
    probsize=geometry(Nel,float(El[-1]))
elif etypes =='G':
#    nB=basis(float(MapX[-1]),MapX[0])
    nB=basis(float(El[-1]),etypes)
    geom=geometry(Nel,float(MapX[-1])) 
    probsize=geometry(Nel,float(El[-1]))
elif etypes=='IF' or etypes == 'IFM':
    geom=geometry(Nel,float(MapX[-1])) 
    probsize=geometry(Nel,float(El[-1]))

Bp = (geom.E1*geom.E2-g(geom.xgam)*(geom.E2-geom.E1)-g(geom.L)*geom.E1)/(geom.E2*(geom.xgam*(geom.E2-geom.E1)+geom.L*geom.E1))

if etypes =='M' or etypes =='L':
    nodes = np.linspace(0.,geom.L,geom.nNNodes)
    elems = np.vstack((np.arange(k,k+nodes.size-1,int(MapX[-1])) for k in range(int(MapX[-1])+1))).T
    globK=0*np.eye((B.enrich)*probsize.nNNodes)                                # This changes because now we no longer have iso-p map
    globF=np.zeros((B.enrich)*probsize.nNNodes)    
    prescribed_dof=np.array([[0,0],
                             [-B.enrich,1]])
elif etypes=='G':
    nodes = np.linspace(0.,probsize.L,probsize.nNNodes)
    elems = np.vstack((np.arange(k,k+nodes.size-1,int(El[-1])) for k in range(int(El[-1])+1))).T  #Connectivity (changes in case of interface enrichment)
    enrich_nodesidx=np.arange(0,len(nodes),1)[find_nodes(nodes)]                      # index of the nodes to be enriched 
    enrich_nodes=nodes[find_nodes(nodes)]                                      # the actual nodes enriched (identified here)
    
    globK=0*np.eye((B.enrich+1)*probsize.nNNodes)                              # add additional dofs corr. to interface enrichment
    globF=np.zeros((B.enrich+1)*probsize.nNNodes) 
    prescribed_dof=np.array([[0,0],
                             [-B.enrich-1,1]])
    
elif etypes == 'IF' or etypes == 'IFM':
    if Nel % 2 != 1:
        raise Exception('Interface Enrichment not needed, specify odd no. of elements !')
    else:
        nodes = np.linspace(0.,geom.L,geom.nNNodes)
        elems = np.vstack((np.arange(k,k+nodes.size-1,int(El[-1])) for k in range(int(El[-1])+1))).T  #Connectivity (changes in case of interface enrichment)
        enrich_nodesidx=np.arange(0,len(nodes),1)[find_nodes(nodes)]                      # index of the nodes to be enriched 
        enrich_nodes=nodes[find_nodes(nodes)]                                      # the actual nodes enriched (identified here)
#    Now decide the size of globK and globF now that you have figured out enriched nodes 
        globK=0*np.eye((B.enrich+1)*probsize.nNNodes+2*(B.enrich+1))               # add additional dofs corr. to interface enrichment
        globF=np.zeros(len(globK)) 
        prescribed_dof=np.array([[0,0],
                                 [-B.enrich-1,1]])

#for k in range(len(elems)):
    
dof=np.inf*np.ones(len(globK))                                                 # initialize to infinity 
prescribed_forc=0*np.array([[-1,-geom.M*geom.g]])                              # Don't use `int` here! (prescribed concentrated loads) 
dof[prescribed_dof[:,0]]=prescribed_dof[:,1]
#dof[1]=0
for k in range(elems[:,0].size):                                               #Assembly of Global System
    elnodes=elems[k]
    elflag='nn'                                                                #element flag: nn--normal, bll--blending left, blr--blending right, rpp--reproducing
    if etypes=='M' or etypes=='L':
        elemord=int(El[-1])+1
        globdof = np.array([k*(elemord-1)+i for i in range(B.enrich*elemord)],int)
        ndofnode=1
    elif etypes=='G':
        elemord = B.enrich+int(El[-1])
        ndofsel=(int(El[-1])+1)*(B.enrich+1)
        globdof = np.array([k*(ndofsel-B.enrich-1)+i for i in range(ndofsel)],int)
        ndofnode=deepcopy(elemord)
    elif etypes == 'IF' or etypes =='IFM':
        elemord = B.enrich+int(El[-1])
        ndofsel=(int(El[-1])+1)*(B.enrich+1)
        if k+1 < mid_elem_idx:
            globdof=[k*(ndofsel-B.enrich-1)+i for i in range(ndofsel)]             # 1-D list of dofs instead of array just to make sure 
#            print(globdof)
            ndofnode=B.enrich+1
        elif k+1 == mid_elem_idx:
            elflag = 'bll'
            ndofnode=B.enrich+1
            globdof=[k*(ndofsel-B.enrich-1)+i for i in range(ndofsel)]
#            print(globdof)
            globdofend=deepcopy(globdof[-1])
            globdof += [globdofend+i for i in range(1,ndofnode+1)]             # adding the extra dofs at farther node (note phi2 comes here then)
            globdofend=deepcopy(globdof[-1])
        elif k==mid_elem_idx:                                                  # reproducing element 
            ndofnode=B.enrich+1
            elflag='rpp'
            globdof=[globdofend-2*(ndofnode)+1+i for i in range(4*ndofnode)]
            globdofend=deepcopy(globdof[-1])
#            print(globdof)
        elif k-1 == mid_elem_idx:
            ndofnode=B.enrich+1
            elflag='blr'
            globdof=[globdofend-2*(ndofnode)+1+i for i in range(3*ndofnode)] # adding extra dofs at the nearer node (blr)
            globdofend=deepcopy(globdof[-1])
#            print(globdof)
        elif k-1 > mid_elem_idx:
            ndofnode=B.enrich+1
            elflag=='nn'
            globdof=[globdofend-ndofnode+1 + i for i in range(2*ndofnode)]
            globdofend = deepcopy(globdof[-1])
#            print(globdof)
     
    if etypes=='L' or etypes=='M' or etypes=='G' or etypes=='IF' or etypes == 'IFM':                                 # 1D lagrange/legendre/GFEM 
        nodexy=nodes[elnodes]
    else:
        nodexy=nodexy[:,elnodes]                                               #for 2D/3D code (extension)
#    print(globdof)
    globdofarr=np.array(globdof)
#    print(elflag)
    kel,fel=loc_mat(nodexy,elflag,ndofnode)
    globK[np.ix_(globdofarr,globdofarr)] += kel                                      #this process would change in vectorization 
    globF[globdof] += fel

globF[int(prescribed_forc[:,0])] += prescribed_forc[:,1]
#print(globF[-1])
fdof=dof==np.inf                                                               # free dofs
nfdof=np.invert(fdof)                                                          # specified dofs 


if etypes=='L' or etypes=='M':  
    AA=globK[np.ix_(fdof,fdof)]
#    print(nla.cond(globK[np.ix_(fdof,fdof)]))
    bb=globF[fdof]-globK[np.ix_(fdof,nfdof)] @ dof[nfdof]                                                                           
    dofme=nla.solve(AA,bb)
elif etypes=='G' or etypes=='IF' or etypes == 'IFM':
    AA = globK[np.ix_(fdof,fdof)]
    bb = globF[fdof]-globK[np.ix_(fdof,nfdof)] @ dof[nfdof]
    dofme = babuskasolve(AA,bb)
#    AA=sp.csr_matrix( globK[np.ix_(fdof,fdof)])
##    print(nla.cond(globK[np.ix_(fdof,fdof)]))
#    bb=globF[fdof]-globK[np.ix_(fdof,nfdof)] @ dof[nfdof]
#    dofme,infodof = sla.minres(AA,bb,tol=1.e-12)                                 #Use Bi-Conjugate Gradient to Solve Ax=b (?)

dof[fdof]=dofme.copy()
    
if etypes == 'L' or etypes == 'M':
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
    if "".join(El.rsplit(El[-1]))=='L':
        plt.plot(xpl2,uapp,'rx',label=r'Quadratic Lagrange FE')
    elif "".join(El.rsplit(El[-1]))=='M':
        plt.plot(xpl2,uapp,'rx',label=r'p-Hierarchical GFEM: Degree '+str(El[1]))
    plt.tick_params(axis='both',which='both',direction='in',top=True,right=True,labelsize=18)
    plt.title(r'Number of elements: '+str(Nel)+', NGP = '+str(Np),fontsize=20)
    plt.legend(loc=0,fontsize=18)
    plt.grid(True,linestyle='--')
    plt.text(0.1,0.4,r'$U^h$ = '+str(0.5*dof @ (globK @ dof)),transform=ax.transAxes,
             fontsize=22,bbox=dict(facecolor='red', alpha=0.5))
Uh = 0.5*dof @ (globK @ dof)
print('Uh={0:.15f}'.format(Uh))
