import numpy as np
import matplotlib.pyplot as plt

# Define variables
sigma = 0.4
mu = 1
gam = 1
s = 0.3
m_0 = 1
g = 0.032
R = 0.37

# Create space
lx = 5
l = np.min([np.abs(0.1/s),0.1/np.sqrt(gam)])
dx = 0.2*l
nx = int(2*lx/dx + 1)
x = np.linspace(-lx,lx,nx)

mask_V = np.abs(x) < R
mask_in =  np.abs(x) < (l + R)
mask_outer_rim = (np.abs(x) > l + R)*(np.abs(x) < (1.3*l + R))
mask_out = np.abs(x) > (l + R)
mask_inner_rim = (np.abs(x) < (l + R))*(np.abs(x) > (0.7*l + R))
mask_in_more = (np.abs(x) < (1.3*l + R))
mask_out_more = (np.abs(x) > (0.7*l + R))

V = np.zeros(nx)
V[np.abs(x) < R] = -1000

def L2_error(p, pn):
    return np.sqrt(np.sum((p - pn)**2)/np.sum(pn**2))     

def jacobi_u(u,m):
    u[0] = -g*m_0/gam
    u[-1] = -g*m_0/gam
    l2_target = 1e-7
    l2norm = 1
    
    while l2norm > l2_target:
        
        un = u.copy()
        
        un_mask_in = u.copy()
        un_mask_in[mask_outer_rim] = np.exp(-un_mask_in[mask_outer_rim]/(mu*sigma**2))
        
        A_phi = mu*sigma**4/(dx**2) - V[mask_in_more][1:-1] 
        S_phi = 0.5*mu*sigma**4*(un_mask_in[mask_in_more][2:] + un_mask_in[mask_in_more][:-2])/(dx**2) 
 
        u[mask_in] = S_phi[np.abs(x[mask_in_more][1:-1]) < R + l]/A_phi[np.abs(x[mask_in_more][1:-1]) < R + l]
        
        un_mask_out = u.copy()

        un_mask_out[mask_inner_rim] = -mu*sigma**2*np.log( un_mask_out[mask_inner_rim])
        
        A_u = 1 + (sigma/dx)**2
        
        S_u = 0.5*sigma**2*(un_mask_out[2:]+ un_mask_out[:-2])/(dx**2) - (un_mask_out[2:]- un_mask_out[:-2])**2/(8*dx**2*mu) - s*(un_mask_out[2:]- un_mask_out[:-2])/(2*dx) - g*m[1:-1]
        
        u[mask_out] = np.hstack([u[0],S_u[mask_out[1:-1]]/A_u,u[-1]])
        
        l2norm = L2_error(u,un)
        
        print(l2norm)
        
    return u

def jacobi_m(m,u):
    m[0] = m_0
    m[-1] = m_0
    l2_target = 1e-7
    l2norm = 1
    
    un = u.copy()
    
    un[mask_inner_rim] = -mu*sigma**2*np.log(un[mask_inner_rim])
    
    u_x = (un[2:] - un[:-2])/(2*dx)
    u_xx = (un[2:] + un[:-2] - 2*un[1:-1])/(dx**2)
    
    while l2norm > l2_target:
        
        mn = m.copy()
        
        mn_mask_in = m.copy()
        mn_mask_in[mask_outer_rim] = mn[mask_outer_rim]/np.exp(-u[mask_outer_rim]/(mu*sigma**2)) 
        
        A_gamma = mu*sigma**4/(dx**2) - V[mask_in_more][1:-1] 
        S_gamma = 0.5*mu*sigma**4*(mn_mask_in[mask_in_more][2:] + mn_mask_in[mask_in_more][:-2])/(dx**2) 
 
        m[mask_in] = S_gamma[np.abs(x[mask_in_more][1:-1]) < R + l]/A_gamma[np.abs(x[mask_in_more][1:-1]) < R + l]
        
        mn_mask_out = m.copy()
        mn_mask_out[mask_inner_rim] = mn[mask_inner_rim]*u[mask_inner_rim]
        
        A_m = (u_xx/mu - sigma**2/dx**2)
        
        S_m = -0.5*sigma**2*(mn_mask_out[2:] + mn_mask_out[:-2])/dx**2 - (mn_mask_out[2:] - mn_mask_out[:-2])/(2*dx)*(u_x/mu + s)
        
        m[mask_out] = np.hstack([m[0],S_m[mask_out[1:-1]]/A_m[mask_out[1:-1]],m[-1]])
        
        plt.plot(S_m)
        plt.show()
        
        plt.plot(A_m)
        plt.show()
        
        l2norm = L2_error(m,mn)
        
        # print(l2norm)
    
    return m

u_sol = np.zeros(nx) + 0.2
m_sol = np.zeros(nx) + m_0

l2norm = 1

while l2norm > 10e-6:
    
    m_old = m_sol.copy()
    
    u_sol = jacobi_u(u_sol,m_sol)

    m_sol = jacobi_m(m_sol,u_sol)
    
    l2norm = L2_error(m_sol, m_old)
    
    print(l2norm)

m_sol[mask_in] = m_sol[mask_in]*u_sol[mask_in]

plt.figure(figsize = (8,5))

plt.plot(x,m_sol, label = 'jacobi')
plt.plot(x,np.zeros(nx)+m_0, label = 'base')

plt.legend()
plt.show()
