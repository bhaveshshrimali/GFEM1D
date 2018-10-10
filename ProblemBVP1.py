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
from matplotlib.ticker import AutoMinorLocator
from sympy import Symbol,diff,Array,lambdify,legendre,integrate,simplify
plt.rc('text',usetex=True)
xmintick=AutoMinorLocator(20)
ymintick=AutoMinorLocator(20)


def ProblemBVP1(xbval,aVal,NumEl,ElType,MapType,enrch):  

#    Check for any possible enrichments 
#    if enrichment is None and ElType[0] is not 'G': 
#        enrch = 1
#    elif ElType[0] is 'G':
#        enrch=enrichment
#    elif ElType[0] is 'L' or ElType[0] is 'M':
#        enrch=1
#    else: 
#        raise Exception('Either specify enrichment or use lagrange/legendre FE')
    
    class geometry() : #1D geometry parameters    
        def __init__(self,ne,deg):
            self.L=1.
            self.A=1.
            self.E=1.
            self.C=0.
            self.a=aVal
            self.xb=xbval
            self.nLNodes=ne+1                                                      #Linear elements
            self.nQNodes=2*ne+1                                                    #Quadratic elements 
            self.nNNodes=int(ne*deg+1)                                             #Isoparametric deg-th order lagrange elements 
            self.ndim=2                                                            #No. of dofs per node (changes for GFEM isotropic enrichment)
            self.M=0
            self.g=0.
    
    def uex(x,a,xb):
        return (1-x)*(np.arctan(a*(x-xb))+np.arctan(a*xb))
    
    def T(x,a,xb):
        axb = a*(x-xb)
        return 2.*a/(1.+axb**2)+2*(1-x)*a**3/(1+axb**2)**2*(x-xb)
    
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
                self.enrich=enrch
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
                self.enrich=enrch
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
                
    def fexact(x,y,p):
        u,up=y
        a,xb=p
        return [up,1/(geom.E*geom.A)*(geom.C*u-T(x,a,xb))]
    
    def fbc(ya,yb,p):
        a,xb=p
        return np.array([ya[0],yb[0],a-geom.a,xb-geom.xb])
    
    if ElType is None:
        print('No element type specified so using quadratic lagrange')
        El = 'L2'
    else:
        El=ElType                                                                 #1D PoU of degree El[1]
                                                                               
    if MapType is None:
        print('No mapping specified so assuming a linear map from master to physical element')
        MapX='L1'    
    else:
        MapX = MapType
    Np = 90                                                                        #Order of Gauss-Integration for the Discrete Galerkin-Form 
    Nel=NumEl                                                                      #Ask why energy is coming out higher in single precision?
    
    if El[0]=='M' or El[0]=='L':                                                   #1D Mapping function for the physical coordinates (can take only single digits, some modification needed)
        nB=basis(float(MapX[-1]),MapX[0])
        geom=geometry(Nel,float(MapX[-1])) 
        probsize=geometry(Nel,float(El[-1]))
    elif El[0]=='G':
        nB=basis(float(El[-1]),El[0])
        geom=geometry(Nel,float(MapX[-1])) 
        probsize=geometry(Nel,float(El[-1]))
    B=basis(float(El[-1]),El[0])                                                   #Basis for FE fields (isoparametric) 
    #geom=geometry(Nel,float(MapX[-1])) 
    #probsize=geometry(Nel,float(El[-1]))
    GP=GPXi(Np) 
    
    
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
    
    def loc_mat(nodes,a,xb):                                                       #Computes the element quantities when supplied with global nodes
        xi=GP.xi;W=GP.wght
        x=np.array(nB.Ns(xi)).T @ nodes
        Je=np.array(nB.dN(xi)).T @ nodes
    #    print(Je)
        if El[0]=='L' or El[0]=='M':                                               # 1D lagrange or legendre basis
            b1 = np.array(B.Ns(xi)).reshape(-1,1,W.size)                           
            a1 = np.array(B.dN(xi)).reshape(-1,1,W.size)  
            a2=a1.reshape(1,len(a1),-1).copy()
            b2=b1.reshape(1,len(b1),-1).copy()
            a1 *= geom.E/Je*geom.A*W                                               #multiply by weights, jacobian, A, in bitwise fashion 
            b1 *= geom.C*Je*W
            mat=np.tensordot(a1,a2,axes=([1,2],[0,2])) + np.tensordot(b1,b2,axes=([1,2],[0,2]))     #could potentially use einsum to clean it up (but works for now!)
            elemF=(np.array(B.Ns(xi))*W*Je*T(x,a,xb)).sum(axis=1).flatten()
            return mat,elemF
        elif El[0]=='G':                                                           #GFEM approximation 
            halph = nodes[-1] - nodes[0]
            enrch = B.enrich                                                       #enrichment order 
            
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
            elemF = (W*Je*Nf*T(x,a,xb)).sum(axis=-1)
            return mat,elemF.flatten()
     
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
    
    if El[0]=='M' or El[0]=='L':
        nodes = np.linspace(0.,geom.L,geom.nNNodes)
        elems = np.vstack((np.arange(k,k+nodes.size-1,int(MapX[1])) for k in range(int(MapX[1])+1))).T
        globK=0*np.eye((B.enrich)*probsize.nNNodes)                                     # This changes because now we no longer have iso-p map
        globF=np.zeros((B.enrich)*probsize.nNNodes)    
        prescribed_dof=np.array([[0,0],
                                 [-B.enrich,0]])
    elif El[0]=='G':
        nodes = np.linspace(0.,probsize.L,probsize.nNNodes)
        elems = np.vstack((np.arange(k,k+nodes.size-1,int(El[1])) for k in range(int(El[1])+1))).T
        globK=0*np.eye((B.enrich+1)*probsize.nNNodes)                              # This changes because now we no longer have iso-p map
        globF=np.zeros((B.enrich+1)*probsize.nNNodes) 
        prescribed_dof=np.array([[0,0],
                                 [-B.enrich-1,0]])      
        
    
    dof=np.inf*np.ones(len(globK))                                                 # initialize to infinity 
    
    prescribed_forc=0*np.array([[-1,-geom.M*geom.g]])                              # Don't use `int` here! (prescribed concentrated loads) 
    
    dof[prescribed_dof[:,0]]=prescribed_dof[:,1]
    
    for k in range(elems[:,0].size):                                               #Assembly of Global System
        elnodes=elems[k]
        if El[0]=='M' or El[0]=='L':
            elemord=int(El[-1])+1
            globdof = np.array([k*(elemord-1)+i for i in range(B.enrich*elemord)],int)
        elif El[0]=='G':
            elemord = B.enrich+int(El[-1])
            ndofsel=(int(El[-1])+1)*(B.enrich+1)
            globdof = np.array([k*(ndofsel-B.enrich-1)+i for i in range(ndofsel)],int)
    #        globdof = globdof[[0,2,3,1]]
        if El[0]=='L' or El[0]=='M' or El[0]=='G':                                               # 1D lagrange or legendre polynomials 
            nodexy=nodes[elnodes]
        else:
            nodexy=nodexy[:,elnodes]
        kel,fel=loc_mat(nodexy,geom.a,geom.xb)
        globK[np.ix_(globdof,globdof)] += kel                                      #this process would change in vectorization 
        globF[globdof] += fel
    
    globF[int(prescribed_forc[:,0])] += prescribed_forc[:,1]
    #print(globF[-1])
    fdof=dof==np.inf                                                               # free dofs
    nfdof=np.invert(fdof)   
    #globK += 1.e-8*np.eye(len(globK))  
    
                                                                                   # specified dofs 
    #dof[fdof],_=sla.gmres(AA,bb)
    #TAs = np.diag(np.diag(AA)**(-0.5)).copy()
    #print(TAs) 
    #AA = TAs @  AA @ TAs + 1.e-10*np.eye(len(AA))
    #print(TAs)
    if El[0]=='L' or El[0]=='M':  
        AA=globK[np.ix_(fdof,fdof)]
        bb=globF[fdof]-globK[np.ix_(fdof,nfdof)] @ dof[nfdof]                                                                           
        dofme=nla.solve(AA,bb)
    elif El[0]=='G':
        AA=sp.csr_matrix( globK[np.ix_(fdof,fdof)])
        bb=globF[fdof]-globK[np.ix_(fdof,nfdof)] @ dof[nfdof]
        dofme,infodof = sla.bicg(AA,bb,tol=1.e-15)                                 #Use Bi-Conjugate Gradient to Solve Ax=b (?)
    dof[fdof]=dofme.copy()

    xplot=np.linspace(0.,geom.L,1000)                                              #sampling points for the exact solution
        
    #plotfigs(xplot) 
    ue=uex(xplot,geom.a,geom.xb)
    xpl2 = np.linspace(nodes[0],nodes[-1],10*geom.nQNodes-1)
    fittype='vandermonde'
    plot_it=False
    if plot_it:
        if El[0]=='L' or El[0]=='M':
            if fittype=='spline':
                xpl2,uapp=fsample(xpl2,dof[0:len(dof):int(El[1])],nodes[0:len(nodes):2],fittype)
            elif fittype=='vandermonde':
                xpl2,uapp=fsample(xpl2,dof,nodes,fittype)
                
            y_a=np.zeros((2,nodes.size))
            y_b=np.zeros((2,nodes.size))
            
            res_a=solve_bvp(fexact,fbc,nodes,y_a,p=[geom.a,geom.xb])
            
            plt.figure(figsize=(10,10))
            ax=plt.gca()
            ax.yaxis.set_minor_locator(ymintick)
            ax.xaxis.set_minor_locator(xmintick)
            
            plt.text(0.35,0.33,r'$a=$ '+str(geom.a),transform=ax.transAxes,
                     fontsize=22,bbox=dict(facecolor='blue', alpha=0.5))
            if El[0]=='L':
                plt.plot(xpl2,uapp,'rx',label=r'Quadratic Lagrange FE')
            elif El[0]=='M' or El[0]=='G':
                plt.plot(xpl2,uapp,'rx',label=r'p-Hierarchical FEM: Degree '+str(El[1]))
            plt.plot(xplot,res_a.sol(xplot)[0],color='black',label=r'Exact Solution')
            
            plt.tick_params(axis='both',which='both',direction='in',top=True,right=True,labelsize=20)
            plt.title(r'Number of elements: '+str(Nel)+', NGP = '+str(Np),fontsize=20)
            plt.legend(loc=0,fontsize=18)
            ax.tick_params(which='major',length=6)
            ax.tick_params(which='minor',length=3)
            
            plt.grid(True,linestyle='--')
            plt.text(0.25,0.2,r'$U^h$ = '+str(0.5*dof @ (globK @ dof)),transform=ax.transAxes,
                     fontsize=22,bbox=dict(facecolor='red', alpha=0.5))
    
    if geom.a==0.5:
        Uexact=0.0408777548
    elif geom.a==50:
        Uexact=25.138142063
    Uh = 0.5*dof @ (globK @ dof)
    errUh = (abs(Uh-Uexact)/Uexact)**0.5
    print(errUh)
    print(Uh)
    return Uh,errUh,dof.size

if __name__ == '__main__':
    xbval=0.2
    aVal=0.5
    NumEl=4
    ElType='G1'
    MapType='L1'
    enrch=1
    ProblemBVP1(xbval,aVal,NumEl,ElType,MapType,enrch)
#    