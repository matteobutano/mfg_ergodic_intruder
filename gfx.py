import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

#  Choose options

save = 0
mode = "density"

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Computer Modern Serif"
})

with open('config.json') as f:
    var = json.loads(f.read())

usual = mpl.cm.hot_r(np.arange(256))
saturate = np.ones((int(256/100),4))

for i in range(3):
    saturate[:,i] = np.linspace(usual[-1,i],0,saturate.shape[0])

cmap = np.vstack((usual,saturate))
cmap = mpl.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

#Main Parameters
xi = var['mfg_params']['xi']
c_s = var['mfg_params']['c_s']
Lx = var['room']['lx']
Ly = var['room']['ly']
Nx = var['room']['nx']
Ny = var['room']['ny']

# Constants
R = var['room']['R']
s = var['room']['s']
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

# Function plotting 

def im(m,d):   
    plt.figure(figsize=(10,10))
    plt.ylim((-d,d))
    plt.xlim((-d,d))
    plt.title('s={} c_s={} R={} xi={} '.format(round(s,2),round(c_s,2),R,round(xi,2)),size= 20)
    plt.xticks([-d,0,d],[-d,0,d],size = 20)
    plt.yticks([-d,0,d],[-d,0,d],size = 20)
    a = plt.arrow(0,-0.2,0,0.25,width = 0.1,head_width = .3,head_length = 0.2,color = 'black',zorder= 10)
    c = plt.Circle((0, 0),radius = R)
    plt.gca().add_artist(a)
    plt.gca().add_artist(c)
    plt.imshow(m,extent=[-Lx,Lx,-Ly,Ly],cmap = cmap)
    plt.colorbar()
    if save == 1:
        plt.savefig('figs/m_Nx='+str(Nx)+'_Ny='+str(Ny)+'_Lx='+str(Lx)+'_Ly='+str(Ly)+'_xi='+str(xi)+'_c_s='+str(c_s)+'.png')
    plt.show()
    plt.close()
    
# Function plotting velocitites, plotted every l grid points (for visibility), in a box of side d and using desnity m to create transparency

def quiv(ax,ay,l,d,m):
    x = X[1:-1,1:-1][::l,::l]
    y = Y[1:-1,1:-1][::l,::l]
    mtr = m[1:-1,1:-1]
    mtr = mtr[::l,::l]
    mtr = (mtr-np.min(mtr))/np.max(mtr)
    ax = ax[::l,::l]
    ay = ay[::l,::l]
    plt.figure(figsize=(10,10))
    plt.axis('equal')
    plt.ylim((-d,d))
    plt.xlim((-d,d))
    plt.title('s={} c_s={} R={} xi={} '.format(round(s,2),round(c_s,2),R,round(xi,2)),size= 20)
    plt.xticks([-d,0,d],[-d,0,d],size = 50)
    plt.yticks([-d,0,d],[-d,0,d],size = 50)
    plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    a = plt.arrow(0,-0.2,0,0.25,width = 0.1,head_width = .3,head_length = 0.2,color = 'black',zorder= 10)
    c = plt.Circle((0, 0),radius = R)
    plt.gca().add_artist(a)
    plt.gca().add_artist(c)
    plt.quiver(x,y,ax,ay+s,angles='xy', scale_units='xy', scale=1, pivot = 'mid', alpha = mtr)
    if save == 1:
        plt.savefig('figs/v_Nx='+str(Nx)+'_Ny='+str(Ny)+'_Lx='+str(Lx)+'_Ly='+str(Ly)+'_xi='+str(xi)+'_c_s='+str(c_s)+'.png')
    plt.show()
    plt.close()
 
# Upload data about density m and velocities vx and vy            

m = np.genfromtxt('data/m_Nx='+str(Nx)+'_Ny='+str(Ny)+'_Lx='+str(Lx)+'_Ly='+str(Ly)+'_xi='+str(xi)+'_c_s='+str(c_s)+'.txt')
vx = np.genfromtxt('data/vx_Nx='+str(Nx)+'_Ny='+str(Ny)+'_Lx='+str(Lx)+'_Ly='+str(Ly)+'_xi='+str(xi)+'_c_s='+str(c_s)+'.txt')
vy = np.genfromtxt('data/vy_Nx='+str(Nx)+'_Ny='+str(Ny)+'_Lx='+str(Lx)+'_Ly='+str(Ly)+'_xi='+str(xi)+'_c_s='+str(c_s)+'.txt')

# Uncomment here to have velocity plots

if mode == "density":
    
    im(m,Lx)
    
elif mode == "velocity":
    
    quiv(vx,vy,2,2,m)

elif mode == "perpendicular":

    plt.plot(x,m[Ny//2,:],label="perpendicular")
    plt.legend()
    plt.show()
    
elif mode =="parallel":
    plt.plot(y,m[:,Nx//2],label ="parallel")
    plt.legend()
    plt.show()
    
