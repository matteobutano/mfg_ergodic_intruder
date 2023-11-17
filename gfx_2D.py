import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

#  Choose options

save = 0
mode = "density"

# Enable LaTeX
plt.rcParams['text.usetex'] = True

# Set the font to be serif, e.g., Times
plt.rcParams['font.family'] = 'serif'


with open('../config.json') as f:
    var = json.loads(f.read())

usual = mpl.cm.hot_r(np.arange(256))
saturate = np.ones((int(256/3),4))

for i in range(3):
    saturate[:,i] = np.linspace(usual[-1,i],0,saturate.shape[0])

cmap = np.vstack((usual,saturate))
cmap = mpl.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

# Main Parameters
xi = var['mfg_params']['xi']
c_s = var['mfg_params']['c_s']

# Constants
R = var['room']['R']
s = var['room']['s']
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
dx = 0.1*l
dy = dx
nx = int(2*lx/dx + 1)
ny = int(2*ly/dy + 1)
x = np.linspace(-lx,lx,nx)
y = np.linspace(-ly,ly,ny)

X,Y = np.meshgrid(x,y)

def norm(u,v):
    return np.sqrt(u**2+v**2)

# Function plotting 

def im(m,d):   
    plt.figure(figsize=(10,10))
    plt.ylim((-d,d))
    plt.xlim((-d,d))
    # plt.title(fr'$s$={s:.2f} $c_s$={c_s:.2f} $R$={R:.2f} $\xi$={xi:.2f} $\gamma$={gam:.2f}',size= 20)
    # plt.xticks([-d,0,d],[-d,0,d],size = 20)
    # plt.yticks([-d,0,d],[-d,0,d],size = 20)
    plt.axis('off')
    a = plt.arrow(0,-0.2,0,0.25,width = 0.1,head_width = .3,head_length = 0.2,color = 'black',zorder= 10)
    c = plt.Circle((0, 0),radius = R)
    scale = plt.arrow(0,-1.2,1,0,width = 0.1,head_width = 0,head_length = 0,color = 'green')
    plt.gca().add_artist(a)
    plt.gca().add_artist(c)
    plt.gca().add_artist(scale)
    plt.imshow(np.flip(m,axis = 0),extent=[-lx,lx,-ly,ly],cmap = cmap)
    plt.clim(0,6)
    # plt.colorbar()
    if save == 1:
        plt.savefig('figs/m_m_0='+str(m_0)+'_lx='+str(lx)+'_ly='+str(ly)+'_xi='+str(round(xi,2))+'_c_s='+str(round(c_s,2))+'_gamma='+str(round(gam,2))+'.png')
    plt.show()
    plt.close()
    
# Function plotting velocities, plotted every l grid points (for visibility), in a box of side d and using desnity m to create transparency

def quiv(ax,ay,l,d,m):
    x = X[1:-1,1:-1][::l,::l]
    y = Y[1:-1,1:-1][::l,::l]
    mtr = m[1:-1,1:-1]
    mtr = mtr[::l,::l]
    mtr = (mtr-np.min(mtr))/np.max(mtr)
    ax = ax[::l,::l]
    ay = ay[::l,::l]
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.ylim((-d,d))
    plt.xlim((-d,d))
    # plt.title(fr'$s$={s:.2f} $c_s$={c_s:.2f} $R$={R:.2f} $\xi$={xi:.2f}',size= 20)
    plt.xticks([-d,0,d],[-d,0,d],size = 50)
    plt.yticks([-d,0,d],[-d,0,d],size = 50)
    plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    a = plt.arrow(0,-0.2,0,0.25,width = 0.1,head_width = .3,head_length = 0.2,color = 'black',zorder= 10)
    c = plt.Circle((0, 0),radius = R)
    plt.gca().add_artist(a)
    plt.gca().add_artist(c)
    plt.quiver(x,y,ax,ay + s,angles='xy', scale_units='xy', scale=1, pivot = 'mid', alpha = mtr)
    if save == 1:
        plt.savefig('figs/v_m_0='+str(m_0)+'_lx='+str(lx)+'_ly='+str(ly)+'_xi='+str(round(xi,2))+'_c_s='+str(round(c_s,2))+'_gamma='+str(round(gam,2))+'.png')
    plt.show()
    plt.close()
 
# Upload data about density m and velocities vx and vy            

m = np.genfromtxt('../data/m_m_0='+str(m_0)+'_lx='+str(lx)+'_ly='+str(ly)+'_xi='+str(round(xi,2))+'_c_s='+str(round(c_s,2))+'_gamma='+str(round(gam,2))+'.txt')
vx = np.genfromtxt('../data/vx_m_0='+str(m_0)+'_lx='+str(lx)+'_ly='+str(ly)+'_xi='+str(round(xi,2))+'_c_s='+str(round(c_s,2))+'_gamma='+str(round(gam,2))+'.txt')
vy = np.genfromtxt('../data/vy_m_0='+str(m_0)+'_lx='+str(lx)+'_ly='+str(ly)+'_xi='+str(round(xi,2))+'_c_s='+str(round(c_s,2))+'_gamma='+str(round(gam,2))+'.txt')

# Uncomment here to have velocity plots

if mode == "density":
    
    im(m,1.5)
    
elif mode == "velocity":
    
    quiv(vx,vy,6,1.5,m)

elif mode == "perpendicular":

    plt.plot(x,m[ny//2,:],label="perpendicular")
    plt.legend()
    plt.show()
    
elif mode =="parallel":
    plt.plot(y,m[:,nx//2],label ="parallel")
    plt.legend()
    plt.show()
    
