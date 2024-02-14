#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import json

# Main Parameters

with open(r'mod_config.json') as f:
    var = json.loads(f.read())
     
Lx = var['room']['lx']
Ly = var['room']['ly']


dx = var['room']['dx']
dy = dx
Nx = int(2*Lx/dx + 1)
Ny = int(2*Ly/dy + 1)

# Constants
m_0 = 1
mu = 1
V = -10e10
g = -0.1
sigma = 0.2
lam = -g*m_0 

#Define grid 
dx = (2*Lx)/(Nx-1)
dy = (2*Ly)/(Ny-1)
x = np.linspace(-Lx,Lx,Nx)
y = np.linspace(-Ly,Ly,Ny)
X,Y = np.meshgrid(x,y)

x_0 = -5
y_0 = -5

def L2_error(p, pn):
    return np.sqrt(np.sum((p - pn)**2)/np.sum(pn**2))       

C = np.zeros((Ny,Nx))
# C[np.sqrt((X - x_0)**2 + (Y - y_0)**2) < R] = V

# C[(np.abs(X) < 8) & (np.abs(Y + 6) < 0.5 ) ] = V

# Main algorithm 
def jacobi(m,p,im):
    p[:,0]  = 0
    p[0,:]  = 0
    p[-1,:] = 0
    p[:,-1] = 0
    p[0,np.abs(x-0) < 2] = 5
    # p[np.abs(y) < 3,0] = 1
    l2_target = 1e-7
    l2norm = 1    
    while l2norm > l2_target:
      
        pn = p.copy()
        A = -2*mu*sigma**4/(dx*dy) + lam + g*m[1:-1,1:-1] + C[1:-1,1:-1]
        Q = pn[1:-1,2:] + pn[1:-1, :-2] + pn[2:, 1:-1] + pn[:-2, 1:-1]
        S = -(0.5*mu*Q*sigma**4)/(dx*dy)
        p[1:-1,1:-1] = S/A
        l2norm = L2_error(p,pn)
        
        print(l2norm)
        
    return p

def v(p):
    
    vy = (p[2:, 1:-1] - p[:-2, 1:-1])/(2*dx)
    vx = (p[1:-1,2:] - p[1:-1, :-2])/(2*dy)
    
    return vx,vy

p_0 = np.zeros((Ny,Nx)) + 1
m = np.full((Ny,Nx),1)
p = jacobi(m,p_0,False)

vx,vy = v(p)

C_mask = np.zeros((Ny,Nx)) + 1
C_mask[C!= 0] = 0


# l = 2
# plt.quiver(X[1:-1,1:-1][::l,::l], Y[1:-1,1:-1][::l,::l], C_mask[1:-1,1:-1][::l,::l]*vx[::l,::l]/np.abs(vx[::l,::l]), C_mask[1:-1,1:-1][::l,::l]*vy[::l,::l]/np.abs(vy[::l,::l]))
# plt.show()

plt.pcolor(X,Y,p)
plt.clim(0,m_0*1.5)
plt.colorbar()
plt.show()



