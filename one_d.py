import numpy as np
import matplotlib.pyplot as plt


lx = 2
dx = 0.01
nx = int(lx/dx + 1)
x = np.linspace(-lx, lx, nx)

C = np.zeros(nx)
C[np.abs(x)< 1] = 10000


def L2_error(p, pn):
    return np.sqrt(np.sum((p - pn)**2)/np.sum(pn**2)) 

def jacobi_1D(p):
    
    p[0] = 1
    p[-1] = 3
    l2_error = 1
    while l2_error > 10e-6:
        
        pn = p.copy()
        p[1:-1] = 0.5*(pn[2:] + pn[:-2])/(1 + C[1:-1]*dx**2)
        
        l2_error = L2_error(p,pn)
        print(l2_error)
        
    return p

p = np.zeros(nx) + 1
p = jacobi_1D(p)

plt.plot(x,p)
plt.show()
        