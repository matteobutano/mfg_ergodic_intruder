import numpy as np
import matplotlib.pyplot as plt


lx = 1
dx = 0.1
nx = int(lx/dx + 1)
x = np.linspace(-lx, lx, nx)

k = -1


def L2_error(p, pn):
    return np.sqrt(np.sum((p - pn)**2)/np.sum(pn**2)) 

def jacobi_1(p):
    
    p[0] = 0.5
    
    p[-1] = 0
    
    l2_error = 1
    
    while l2_error > 10e-8: 
        pn = p.copy()

        p[1:-1] = 0.5*(pn[2:] + pn[:-2])
        p[1:-1] = p[1:-1] + ( p[0]  + k*dx - p[1] )
        #  p[-1] = (p[-2] - p[-3]) + p[-2]
        
        l2_error = L2_error(p,pn)
        print(l2_error)
        
    return p

p = np.zeros(nx) + 5
p = jacobi_1(p)

plt.plot(x,p)
plt.show()
        