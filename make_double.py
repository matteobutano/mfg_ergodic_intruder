import mfg_ergodic_intruder.mfg_ergodic as mfg
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

fontsize = 30
clim = 1.5

# Enable LaTeX
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Create colormap
usual = mpl.cm.hot_r(np.arange(256))
saturate = np.ones((1,4))

for i in range(3):
    saturate[:,i] = np.linspace(usual[-1,i],0,saturate.shape[0])

cmap = np.vstack((usual,saturate))
cmap = mpl.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])


m_small = mfg.mfg('bottom_left_small')
m_large = mfg.mfg('bottom_left_large')

print(m_small.gamma_1, m_small.gamma_2)
print(m_large.gamma_1, m_large.gamma_2)


print('R_tilde', m_small.R/m_small.xi, 's_tilde', m_small.s/m_small.c_s)

m_small.draw_density(clim = clim, save = True, axis = False, title = False, colorbar = False, scale = False)
m_large.draw_density(clim = clim, save = True, axis = False, title = False, colorbar = False, scale = False)

im = plt.imshow(m_small.m,cmap = cmap)
im.set_clim(0,clim)

fig, axs = plt.subplots(1,2,figsize = (12,6))

m_small_im = plt.imread('gfx/bottom_left_small.png')
m_large_im = plt.imread('gfx/bottom_left_large.png')


axs[0].imshow(m_small_im)
axs[0].axis('off')
axs[0].set_title(r' $(\tilde \gamma^{(1)} << 1,\;\tilde \gamma^{(2)} << 1)$',fontsize = fontsize)
axs[1].imshow(m_large_im)
axs[1].set_title(r' $(\tilde \gamma^{(1)} >> 1,\;\tilde \gamma^{(2)} >> 1)$',fontsize = fontsize)
axs[1].axis('off')

cbar = fig.colorbar(im, ax = axs, orientation = 'horizontal', shrink = 0.9)
cbar.set_ticks([0,0.5,1,clim])
cbar.ax.tick_params(labelsize = fontsize)



