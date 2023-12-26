from mfg_ergodic_intruder import mfg_ergodic
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import Circle,Arrow

# Load data
m_large = mfg_ergodic.mfg('top_right_large')
m_small = mfg_ergodic.mfg('top_right_small')

# Graphical Parameters 
fontsize = 25
d = 2.2
clim = 1.5

# Enable LaTeX
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Create colormap
usual = mpl.cm.hot_r(np.arange(256))
saturate = np.ones((int(256/256),4))

for i in range(3):
    saturate[:,i] = np.linspace(usual[-1,i],0,saturate.shape[0])

cmap = np.vstack((usual,saturate))
cmap = mpl.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

fig,axs = plt.subplots(1,2,figsize = (14,10))

# Define scales
c_small = Circle((0, 0),radius = m_small.R)
small_l_s = Arrow(0.1, 0, 0,m_small.R + (m_small.s*m_small.xi/m_small.c_s),width= 0.5,color = 'purple',label = r'$l_s$')
small_d_s = Arrow(-0.1, 0, 0, m_small.R + m_small.s/m_small.gam,width= 0.5, color = 'blue',label = r'$d_s$')
c_large = Circle((0, 0),radius = m_large.R)
large_l_s = Arrow(0.1, 0, 0,m_large.R + (m_large.s*m_large.xi/m_large.c_s),width= 0.5,color = 'purple',label = r'$l_s$')
large_d_s = Arrow(-0.1, 0, 0, m_large.R + m_large.s/m_large.gam,width= 0.5, color = 'blue',label = r'$d_s$')

# Left Plot
small_ax = axs[0].imshow(np.flip(m_small.m,axis = 0),extent=[-m_small.lx,m_small.lx,-m_small.ly,m_small.ly],cmap = cmap)
axs[0].add_patch(c_small)
axs[0].add_patch(small_l_s)
axs[0].add_patch(small_d_s)
axs[0].set_title(rf'$\tilde R >> 1\; \tilde s >> 1 \; \gamma ={m_small.gam} $',fontsize = fontsize)
axs[0].set_xticks([-d,0,d],[-d,0,d],size = fontsize)
axs[0].set_yticks([-d,0,d],[-d,0,d],size = fontsize)
axs[0].set_xlim(-d,d)
axs[0].set_ylim(-d,d)
axs[0].legend(fontsize = fontsize)
small_ax.set_clim(0,clim)

# Right Plot
large_ax = axs[1].imshow(np.flip(m_large.m,axis = 0),extent=[-m_large.lx,m_large.lx,-m_large.ly,m_large.ly],cmap = cmap)
axs[1].add_patch(c_large)
axs[1].add_patch(large_l_s)
axs[1].add_patch(large_d_s)
axs[1].set_title(rf'$\tilde R >> 1\; \tilde s >> 1 \; \gamma ={m_large.gam} $',fontsize = 25)
axs[1].set_xticks([-d,0,d],[-d,0,d],size = fontsize)
axs[1].set_yticks([-d,0,d],[-d,0,d],size = fontsize)
axs[1].set_xlim(-d,d)
axs[1].set_ylim(-d,d)
axs[1].legend(fontsize = fontsize)
large_ax.set_clim(0,clim)

# Colorbar Parameters
cbar = fig.colorbar(small_ax,ax = axs,orientation = 'horizontal',shrink = 0.8)
cbar.set_ticks([0,0.5,1,clim])
cbar.set_label(r'Pedestrian Density ped/m²',fontsize = fontsize)
cbar.ax.tick_params(labelsize = fontsize)

plt.show()