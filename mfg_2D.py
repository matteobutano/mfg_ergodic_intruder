#!/usr/bin/python

import numpy as np
import time 
import json
# import matplotlib.pyplot as plt

save = 1

# Read parameters

with open('config.json') as f:
    var = json.loads(f.read())

# Main Parameters
xi = var['mfg_params']['xi']
c_s = var['mfg_params']['c_s']

# Constants
R = var['room']['R']
s = -var['room']['s']
m_0 = var['room']['m_0']
mu = var['mfg_params']['mu']
V = var['mfg_params']['V']
gam = var['mfg_params']['gam']

g = -(2*c_s**2)/m_0
sigma = np.sqrt(2*xi*c_s)

# Create space
lx = var['room']['lx']
ly = var['room']['ly']
l = np.min([np.abs(0.1/s),0.1/np.sqrt(gam)])
dx = 0.2*l
dy = dx
nx = int(2*lx/dx + 1)
ny = int(2*ly/dy + 1)
x = np.linspace(-lx,lx,nx)
y = np.linspace(-ly,ly,ny)

X,Y = np.meshgrid(x,y)

mask_V = np.sqrt(X**2 + Y**2) < R
mask_in =  np.sqrt(X**2 + Y**2) < (l + R)
mask_outer_rim = (np.sqrt(X**2 + Y**2) > l + R)*(np.sqrt(X**2 + Y**2) < (1.3*l + R))
mask_out = np.sqrt(X**2 + Y**2) > (l + R)
mask_inner_rim = (np.sqrt(X**2 + Y**2) < (l + R))*(np.sqrt(X**2 + Y**2) > (0.7*l + R))
mask_in_more = (np.sqrt(X**2 + Y**2) < (1.3*l + R))
mask_out_more = (np.sqrt(X**2 + Y**2) > (0.7*l + R))

V = np.zeros((ny,nx))
V[np.sqrt(X**2 + Y**2) < R] = -1000

def L2_error(p, pn):
    return np.sqrt(np.sum((p - pn)**2)/np.sum(pn**2))     



def jacobi_u(u,m):

    u[0,:] = -g*m_0/gam
    u[:,0] = -g*m_0/gam
    u[-1,:] = -g*m_0/gam
    u[:,-1] = -g*m_0/gam
    
    l2_target = 1e-7
    l2norm = 1
    
    while l2norm > l2_target:
        
        un = u.copy()
    
        un_mask_in = u.copy()
        un_mask_in[mask_outer_rim] = np.exp(-un_mask_in[mask_outer_rim]/(mu*sigma**2))

        A_phi = 2*mu*sigma**4/(dx*dy) - V[1:-1,1:-1]
        S_phi = 0.5*mu*sigma**4*(un_mask_in[2:,1:-1] + un_mask_in[:-2,1:-1] + un_mask_in[1:-1,2:] + un_mask_in[1:-1,:-2])/(dx*dy) 
 
        u[mask_in] = S_phi[mask_in[1:-1,1:-1]]/A_phi[mask_in[1:-1,1:-1]]
    
        un_mask_out = u.copy()        
        un_mask_out[mask_inner_rim] = -mu*sigma**2*np.log(un_mask_out[mask_inner_rim])

        un_xx = un_mask_out[2:,1:-1]+ un_mask_out[:-2,1:-1] + un_mask_out[1:-1, 2:]+ un_mask_out[1:-1,:-2]
        un_y = un_mask_out[2:,1:-1] - un_mask_out[:-2,1:-1]
        un_x = un_mask_out[1:-1,2:] - un_mask_out[1:-1, :-2]
        
        A_u = gam + 2*sigma**2/(dx*dy)
        S_u = 0.5*sigma**2*(un_xx)/(dx*dy) - (un_x**2 + un_y**2)/(8*dx**2*mu) - s*(un_y)/(2*dx) - g*m[1:-1,1:-1]
        
        u[1:-1,1:-1][mask_out[1:-1,1:-1]] = S_u[mask_out[1:-1,1:-1]]/A_u
     
        l2norm = L2_error(u,un)
        
        # print(l2norm)
        
    return u

def jacobi_m(m,u):
    
    m[0,:] = m_0
    m[:,0] = m_0
    m[-1,:] = m_0
    m[:,-1] = m_0
    
    l2_target = 1e-7
    l2norm = 1
    
    un = u.copy()
    
    un[mask_inner_rim] = -mu*sigma**2*np.log(un[mask_inner_rim])
    
    un_xx = (un[2:,1:-1]+ un[:-2,1:-1] + un[1:-1, 2:]+ un[1:-1,:-2] - 4*un[1:-1,1:-1])/(dx*dy)
    un_xx = un_xx*mask_out[1:-1,1:-1]
    
    un_y = (un[2:,1:-1] - un[:-2,1:-1])/(2*dy)
    un_y = un_y*mask_out[1:-1,1:-1]
    
    un_x = (un[1:-1,2:] - un[1:-1, :-2])/(2*dx)
    un_x = un_x*mask_out[1:-1,1:-1]
       
    while l2norm > l2_target:
        
        mn = m.copy()
    
        mn_mask_in = mn.copy()
        mn_mask_in[mask_outer_rim] = mn[mask_outer_rim]/np.exp(-u[mask_outer_rim]/(mu*sigma**2)) 

        A_gamma = 2*mu*sigma**4/(dx*dy) - V[1:-1,1:-1]
        S_gamma = 0.5*mu*sigma**4*(mn_mask_in[2:,1:-1] + mn_mask_in[:-2,1:-1] + mn_mask_in[1:-1,2:] + mn_mask_in[1:-1,:-2])/(dx*dy)
 
        m[mask_in] = S_gamma[mask_in[1:-1,1:-1]]/A_gamma[mask_in[1:-1,1:-1]]
                
        mn_mask_out = mn.copy()
        mn_mask_out[mask_inner_rim] = mn[mask_inner_rim]*u[mask_inner_rim]
        
        mn_x = (mn_mask_out[1:-1,2:] - mn_mask_out[1:-1, :-2])/(2*dx)
        mn_y = (mn_mask_out[2:,1:-1] - mn_mask_out[:-2,1:-1])/(2*dy)
        mn_xx = (mn_mask_out[2:,1:-1]+ mn_mask_out[:-2,1:-1] + mn_mask_out[1:-1, 2:]+ mn_mask_out[1:-1,:-2])/(dx*dy)
        
        A_m = 2*sigma**2/(dx*dy) - un_xx/mu
        S_m = 0.5*sigma**2*mn_xx + (un_x*mn_x + un_y*mn_y)/mu + s*mn_y
        
        m[1:-1,1:-1][mask_out[1:-1,1:-1]] = S_m[mask_out[1:-1,1:-1]]/A_m[mask_out[1:-1,1:-1]]
     
        l2norm = L2_error(m,mn)
        
        # print(l2norm)
        
    return m

def vel(m,p,q):
    phi_grad_x = (p[1:-1,2:]-p[1:-1,:-2])/(2*dx)
    phi_grad_y = (p[2:,1:-1]-p[:-2,1:-1])/(2*dy)
    gamma_grad_x = (q[1:-1,2:]-q[1:-1,:-2])/(2*dx)
    gamma_grad_y = (q[2:,1:-1]-q[:-2,1:-1])/(2*dy)
    v_x = sigma**2*(q[1:-1,1:-1]*phi_grad_x-p[1:-1,1:-1]*gamma_grad_x)/(2*m[1:-1,1:-1])
    v_y = sigma**2*(q[1:-1,1:-1]*phi_grad_y-p[1:-1,1:-1]*gamma_grad_y)/(2*m[1:-1,1:-1])-s
    return np.array([v_x,v_y])

u_sol = np.zeros((ny,nx)) - g*m_0/gam
m_sol = np.zeros((ny,nx)) + m_0

l2norm = 1
tic  = time.time()
alpha = 0.5

print('Computation begins')

while l2norm > 10e-8:
    
    m_old = m_sol.copy()
    
    u_sol = jacobi_u(u_sol,m_sol)

    m_sol = jacobi_m(m_sol,u_sol)
    
    m_sol = alpha*m_sol + (1-alpha)*m_old
    
    l2norm = L2_error(m_sol, m_old)
    
    toc = time.time()
    
    print(f'Error = {l2norm:.3e} Time = {(toc-tic)//3600:.0f}h{((toc-tic)//60)%60:.0f}m{(toc-tic)%60:.0f}s')
    
print('Computation ends')

# p = u_sol.copy()
# p[mask_out] = np.exp(-p[mask_out]/(mu*sigma**2))
# q = m_sol.copy()
# q[mask_out] = q[mask_out]/p[mask_out]
m = m_sol
m[mask_in] = m[mask_in]*u_sol[mask_in]

# v = vel(m_sol,p,q)
# vx,vy = v[0],v[1] 
    
if save == 1:
    np.savetxt(r'data/m_m_0='+str(m_0)+'_lx='+str(lx)+'_ly='+str(ly)+'_xi='+str(round(xi,2))+'_c_s='+str(round(c_s,2))+'_gamma='+str(round(gam,2))+'.txt',m)
    # np.savetxt(r'data/vx_m_0='+str(m_0)+'_lx='+str(lx)+'_ly='+str(ly)+'_xi='+str(round(xi,2))+'_c_s='+str(round(c_s,2))+'_gamma='+str(round(gam,2))+'.txt',vx)
    # np.savetxt(r'data/vy_m_0='+str(m_0)+'_lx='+str(lx)+'_ly='+str(ly)+'_xi='+str(round(xi,2))+'_c_s='+str(round(c_s,2))+'_gamma='+str(round(gam,2))+'.txt',vy)


