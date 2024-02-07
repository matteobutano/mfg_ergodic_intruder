#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import json

# Main Parameters

with open(r'mod_config.json') as f:
    var = json.loads(f.read())
     
xi = var['mfg_params']['xi']
c_s = var['mfg_params']['c_s']
Lx = var['room']['lx']
Ly = var['room']['ly']


dx = var['room']['dx']
dy = dx
Nx = int(2*Lx/dx + 1)
Ny = int(2*Ly/dy + 1)

# Constants
R = var['room']['R']
m_0 = var['room']['m_0']
mu = var['mfg_params']['mu']
V = var['mfg_params']['V']
g = -(2*c_s**2)/m_0
sigma = np.sqrt(2*xi*c_s)
lam = -g*m_0 

#Define grid 
dx = (2*Lx)/(Nx-1)
dy = (2*Ly)/(Ny-1)
x = np.linspace(-Lx,Lx,Nx)
y = np.linspace(-Ly,Ly,Ny)
X,Y = np.meshgrid(x,y)



def norm(u,v):
    return np.sqrt(u**2+v**2)

# plt.imshow(C)

def L2_error(p, pn):
    return np.sqrt(np.sum((p - pn)**2)/np.sum(pn**2))       

# Main algorithm 
def jacobi(m,p):
    p[:,0]  = 0
    p[0,:]  = 0
    p[-1,:] = 0
    p[:,-1] = 0
    l2_target = 1e-7
    l2norm = 1    
    while l2norm > l2_target:
      
        p_more = np.zeros((Ny+2, Nx+2))
        p_more[1:-1,1:-1] = p.copy()
        p_more[0,np.abs(np.hstack([100,x,100])) < 3] = p_more[2,np.abs(np.hstack([100,x,100])) < 3] + 2*dx
        p_more[-1,np.abs(np.hstack([100,x,100])) < 3] = p_more[-3,np.abs(np.hstack([100,x,100])) < 3] + 2*dx

        A = -2*mu*sigma**4/(dx*dy) + lam + (g*m)
        Q = p_more[1:-1,2:] + p_more[1:-1, :-2] + p_more[2:, 1:-1] + p_more[:-2, 1:-1]
        S = -(0.5*mu*Q*sigma**4)/(dx*dy)
        
        p = S/A
        p[0,np.abs(x) < 3] = np.sqrt(m_0)
        p[-1,np.abs(x) < 3] = np.sqrt(m_0)
        
        l2norm = L2_error(p,p_more[1:-1,1:-1])
        
        print(l2norm)
        
        
        # plt.imshow(p**2)
        # plt.colorbar()
        # plt.show()
    return p


p_0 = np.zeros((Ny,Nx)) 
m = np.full((Ny,Nx),m_0)
p = jacobi(m,p_0)


plt.pcolor(X,Y,p**2)
#plt.pcolor(X,Y,C)
# plt.clim(0,m_0*1.5)
plt.colorbar()
plt.show()


