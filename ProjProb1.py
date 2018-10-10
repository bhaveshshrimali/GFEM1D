# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 23:07:15 2018

@author: bshrima2

Master file for ProblemBVP1.py

"""

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import AutoMinorLocator
from itertools import product
from joblib import Parallel,delayed
import multiprocessing as mpl
from ProblemBVP1 import ProblemBVP1 as SolFunc

plt.rc('text',usetex=True)
xmintick=AutoMinorLocator(30)
ymintick=AutoMinorLocator(30)

def processhFEM(ne):
    return SolFunc(xb,a,ne,Els,'L1',enf)

def processpGFEM(enrchmts):
    return SolFunc(xb,a,ne,Els,'L1',enrchmts)

def processpFEM(Et):
    return SolFunc(xb,a,ne,Et,'L1',1)

# Problem 1 -- hversion FEM vs hversion GFEM 
plt.figure(1,figsize=(12,12))
ax=plt.gca()
ax.yaxis.set_minor_locator(ymintick)
ax.xaxis.set_minor_locator(xmintick)
plt.xlabel(r'NDOF',fontsize=22)
plt.ylabel(r'Error in the Energy Norm $\| \cdot \|_E$',fontsize=22)
ax.tick_params(which='major',length=6)
ax.tick_params(which='minor',length=3)



if __name__ == '__main__':
    
    
#    Els = ['L','M','G']                                                            # Element Types
    ElDegs = [str(i) for i in range(1,8)]                                          # Element degrees
    
    #ElTypes = [''.join(x) for x in product(Els,ElDegs)]                            #All possible combinations of element types (not needed though)
#    ElTypes = ['L1,L2,L3,L4,L5,L6,M1,M2,M3,M4,M5,M6']
    
    avals = np.array([0.5,50.])
    a=avals[1]
    xb = 0.2
    ncores=mpl.cpu_count()
    ench=False
    lag_hbasis=True
    lag_pbasis=False
    lag_phbasis=False

#   h-version p-hierarchical GFEM and p-hierarchical FEM
    if lag_hbasis is False and lag_pbasis is False and lag_phbasis is False:
  
        if a==0.5 and ench is not True:
            enf=1
            rlabel='GFEM (Linear Enrichments): $p= $'+str(enf+1)
            Numels=np.array([2**i for i in range(1,6)],int)
            Els='G1'
            resl=np.array([processhFEM(ne) for ne in Numels])
#            with Parallel(n_jobs=ncores) as parll:
#                resl=np.array(parll(delayed(processhFEM)(ne) for ne in Numels))
        elif a==50. and ench is not True:   
            enf=1
            rlabel='GFEM (Linear Enrichments): $p= $'+str(enf+1)
            Numels = 10*np.array([2**i for i in range(6)],int)
            Els='G1'
            resl=np.array([processhFEM(ne) for ne in Numels])
            
#            with Parallel(n_jobs=ncores) as parll:
#                resl=np.array(parll(delayed(processhFEM)(ne) for ne in Numels))
        elif a==0.5 and ench is True:
            rlabel='p-Hierarchical GFEM'
            enrhments = np.arange(1,7,1)
            ne=2
            Els='G1'
            resl=np.array([processpGFEM(enr) for enr in enrhments])
#            with Parallel(n_jobs=ncores) as parll:
#                resl=np.array(parll(delayed(processpGFEM)(enr) for enr in enrhments))
        elif a==50. and ench is True:
            rlabel='p-Hierarchical GFEM'
            ne=5
            Els='G1'
            enrhments = np.arange(1,7,1)
            resl=np.array([processpGFEM(enr) for enr in enrhments])
#            with Parallel(n_jobs=ncores) as parll:
#                resl=np.array(parll(delayed(processpGFEM)(enr) for enr in enrhments))
    elif lag_hbasis:
        if a==0.5:
            Numels = np.array([2**i for i in range(1,6)],int)
            Els = 'L2'
            rlabel=r'h-FEM: Degree = '+str(Els[1])
            resl=np.array([processhFEM(ne) for ne in Numels])
#            with Parallel(n_jobs=ncores) as parll:
#                resl=np.array(parll(delayed(processhFEM)(ne) for ne in Numels))
        elif a==50.:
            Numels = 10*np.array([2**i for i in range(6)],int)
            Els = 'L2'
            rlabel=r'h-FEM: Degree = '+str(Els[1])
            resl=np.array([processhFEM(ne) for ne in Numels])
#            with Parallel(n_jobs=ncores) as parll:
#                resl=np.array(parll(delayed(processhFEM)(ne) for ne in Numels))
    elif lag_pbasis:
        if a==0.5:
            ne=2
            Els = [''.join(x) for x in product(['L'],ElDegs)]
            rlabel=r'p-FEM'
            resl=np.array([processpFEM(El) for El in Els])
#            with Parallel(n_jobs=ncores) as parll:
#                resl=np.array(parll(delayed(processpFEM)(El) for El in Els))
        elif a==50.:
            ne=5
            Els = [''.join(x) for x in product(['L'],ElDegs)]
            rlabel=r'p-FEM'
            resl=np.array([processpFEM(El) for El in Els])
#            with Parallel(n_jobs=ncores) as parll:
#                resl=np.array(parll(delayed(processpFEM)(El) for El in Els))
    elif lag_phbasis:
        if a==0.5:
            ne=2
            Els = [''.join(x) for x in product(['M'],ElDegs)]
            rlabel=r'p-Hierarchical FEM'
            resl=np.array([processpFEM(El) for El in Els])
#            with Parallel(n_jobs=ncores) as parll:
#                resl=np.array(parll(delayed(processpFEM)(El) for El in Els))
        elif a==50.:
            ne=5
            Els = [''.join(x) for x in product(['M'],ElDegs)]
            rlabel=r'p-Hierarchical FEM'
            resl=np.array([processpFEM(El) for El in Els])
#            with Parallel(n_jobs=ncores) as parll:
#                resl=np.array(parll(delayed(processpFEM)(El) for El in Els))
    print(resl[:,0])

    
    plt.text(0.1,0.2,r'$a=$ '+str(a),transform=ax.transAxes,
             fontsize=22,bbox=dict(facecolor='blue', alpha=0.5))
    plt.loglog(resl[:,-1],resl[:,-2],'-s',label=rlabel)
    plt.tick_params(axis='both',which='both',direction='in',top=True,right=True,labelsize=20)
    plt.legend(loc=0,fontsize=22)
    plt.grid(True,linestyle='--')
    
        