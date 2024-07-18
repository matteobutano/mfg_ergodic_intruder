import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import time
import os

class mfg:
    def __init__(self, config, mode = 'read'):
        """
        Create the simulation, initialize the array u and m. 

        Parameters
        ----------
        config : str
            Name of the config file WITHOUT .json.
        mode : str, optional
            Choose if you want to read existing data or to create new. The default is 'read', the other option is "write".

        Returns
        -------
        None.

        """
        self.config = config
        self.mode = mode # read or write
        with open(r'configs/'+ config +'.json') as f:
            var = json.loads(f.read())
        # Parameters of the MFG 
        self.xi      = var['mfg_params']['xi']
        self.c_s    = var['mfg_params']['c_s']
        self.R      = var['room']['R']
        self.s      = var['room']['s']
        self.m_0    = var['room']['m_0']
        self.mu     = var['mfg_params']['mu']
        self.gam    = var['mfg_params']['gam']
        self.g      = -(2*self.c_s**2)/self.m_0
        self.sigma  = np.sqrt(2*self.xi*self.c_s)
        # Create space
        self.lx     = var['room']['lx'] # horizontal half length
        self.ly     = var['room']['ly'] # vertical half length
        self.min_dx = 0.05 # minimal grid step 
        if self.gam > 0:
            self.l    = self.sigma/(10*np.sqrt(2*self.gam))
            self.dx   = np.min([0.2*self.l,self.min_dx])
        else:
            self.lam  = -self.g*self.m_0
            self.dx   = self.min_dx
        self.dy       = self.dx
        self.nx       = int(2*self.lx/self.dx + 1)
        self.ny       = int(2*self.ly/self.dy + 1)
        self.x        = np.linspace(-self.lx,self.lx,self.nx)
        self.y        = np.linspace(-self.ly,self.ly,self.ny)
        self.X,self.Y = np.meshgrid(self.x,self.y)
        
        if mode == 'read':
            if os.path.exists(r'data/m_'+self.config+'.txt'):
                print('Found and read existing data.')
                self.m = np.genfromtxt(r'data/m_'+self.config+'.txt')
                self.vx = np.genfromtxt(r'data/vx_'+self.config+'.txt')
                self.vy = np.genfromtxt(r'data/vy_'+self.config+'.txt')
            else: 
                raise Exception('The data you are looking for does not exist.')
        else:  
            if os.path.exists(r'data/m_'+self.config+'.txt'): 
                raise Exception('This data already exist and cannot be overwritten')
            else: 
                print('No data found, initialising.')
                if self.gam > 0:
                    self.u = np.zeros((self.ny,self.nx)) - self.g*self.m_0/self.gam
                else:
                    self.u = np.zeros((self.ny,self.nx)) + np.sqrt(self.m_0)
                self.m = np.zeros((self.ny,self.nx)) + self.m_0
                self.vx = np.zeros((self.ny-2,self.nx-2))
                self.vy = np.zeros((self.ny-2,self.nx-2))
        V = var['mfg_params']['V']
        self.V = np.zeros((self.ny,self.nx))
        self.V[np.sqrt(self.X**2 + self.Y**2) < self.R] = V
        self.V[:,0]  = V
        self.V[:,-1] = V
        self.gam = var['mfg_params']['gam']
        self.l2_target = 10e-8
        self.alpha = 0.01
        self.verbose = False
         
    def L2_error(self, p, pn):
        return np.sqrt(np.sum((p - pn)**2)/np.sum(pn**2))  
    
    def jacobi_u(self):
        # Create the masks to divide the space for double solution
        mask_in =  np.sqrt(self.X**2 + self.Y**2) < (self.l + self.R)
        mask_outer_rim = (np.sqrt(self.X**2 + self.Y**2) > self.l + self.R)*(np.sqrt(self.X**2 + self.Y**2) < (1.3*self.l + self.R))
        mask_out = np.sqrt(self.X**2 + self.Y**2) > (self.l + self.R)
        mask_inner_rim = (np.sqrt(self.X**2 + self.Y**2) < (self.l + self.R))*(np.sqrt(self.X**2 + self.Y**2) > (0.7*self.l + self.R))
        self.u[0,:] = -self.g*self.m_0/self.gam
        self.u[:,0] = -self.g*self.m_0/self.gam
        self.u[-1,:] = -self.g*self.m_0/self.gam
        self.u[:,-1] = -self.g*self.m_0/self.gam
        l2norm = 1
        i = 1
        while l2norm > self.l2_target:
            un = self.u.copy() # Copy state of u
            un_mask_in = self.u.copy() 
            un_mask_in[mask_outer_rim] = np.exp(-un_mask_in[mask_outer_rim]/(self.mu*self.sigma**2)) # Convert from u to phi near cylinder 
            A_phi = 2*self.mu*self.sigma**4/(self.dx*self.dy) - self.V[1:-1,1:-1]  # Denominator of Jacobi fraction for Cole-Hopf equations
            S_phi = 0.5*self.mu*self.sigma**4*(un_mask_in[2:,1:-1] + un_mask_in[:-2,1:-1] + un_mask_in[1:-1,2:] + un_mask_in[1:-1,:-2])/(self.dx*self.dy)  - self.mu*self.sigma**2*self.s*(un_mask_in[2:,1:-1] - un_mask_in[:-2,1:-1])/(2*self.dy) # Numerator of Jacobi fraction for Cole-Hopf equations
            self.u[mask_in] = S_phi[mask_in[1:-1,1:-1]]/A_phi[mask_in[1:-1,1:-1]] # Modify inner part with values of phi 
            un_mask_out = self.u.copy()        
            un_mask_out[mask_inner_rim] = -self.mu*self.sigma**2*np.log(un_mask_out[mask_inner_rim]) # Convert from phi to u near cylinder
            un_xx = un_mask_out[2:,1:-1]+ un_mask_out[:-2,1:-1] + un_mask_out[1:-1, 2:]+ un_mask_out[1:-1,:-2]
            un_y = un_mask_out[2:,1:-1] - un_mask_out[:-2,1:-1]
            un_x = un_mask_out[1:-1,2:] - un_mask_out[1:-1, :-2]
            A_u = self.gam + 2*self.sigma**2/(self.dx*self.dy) # Denominator of Jacobi fraction for HJB equation
            S_u = 0.5*self.sigma**2*(un_xx)/(self.dx*self.dy) - (un_x**2 + un_y**2)/(8*self.dx**2*self.mu) - self.s*(un_y)/(2*self.dx) - self.g*self.m[1:-1,1:-1] # Numerator of Jacobi fraction for HJB equation
            self.u[1:-1,1:-1][mask_out[1:-1,1:-1]] = S_u[mask_out[1:-1,1:-1]]/A_u
            l2norm = self.L2_error(self.u,un)
            self.u = self.alpha*self.u + (1-self.alpha)*un
            if self.verbose and i%500 == 0:
                print('u error',l2norm)
            i +=1
        
    def jacobi_m(self):
        self.m[0,:] = self.m_0
        self.m[:,0] = self.m_0
        self.m[-1,:] = self.m_0
        self.m[:,-1] = self.m_0
        mask_in =  np.sqrt(self.X**2 + self.Y**2) < (self.l + self.R)
        mask_outer_rim = (np.sqrt(self.X**2 + self.Y**2) > self.l + self.R)*(np.sqrt(self.X**2 + self.Y**2) < (1.3*self.l + self.R))
        mask_out = np.sqrt(self.X**2 + self.Y**2) > (self.l + self.R)
        mask_inner_rim = (np.sqrt(self.X**2 + self.Y**2) < (self.l + self.R))*(np.sqrt(self.X**2 + self.Y**2) > (0.7*self.l + self.R))
        un = self.u.copy()
        un[mask_inner_rim] = -self.mu*self.sigma**2*np.log(un[mask_inner_rim])
        un_xx = (un[2:,1:-1]+ un[:-2,1:-1] + un[1:-1, 2:]+ un[1:-1,:-2] - 4*un[1:-1,1:-1])/(self.dx*self.dy)
        un_xx = un_xx*mask_out[1:-1,1:-1]
        un_y = (un[2:,1:-1] - un[:-2,1:-1])/(2*self.dy)
        un_y = un_y*mask_out[1:-1,1:-1]
        un_x = (un[1:-1,2:] - un[1:-1, :-2])/(2*self.dx)
        un_x = un_x*mask_out[1:-1,1:-1]
        l2norm = 1
        i = 1 
        while l2norm > self.l2_target:
            mn = self.m.copy()
            mn_mask_in = mn.copy()
            mn_mask_in[mask_outer_rim] = mn[mask_outer_rim]/np.exp(-self.u[mask_outer_rim]/(self.mu*self.sigma**2)) 
            A_gamma = 2*self.mu*self.sigma**4/(self.dx*self.dy) - self.V[1:-1,1:-1] 
            S_gamma = 0.5*self.mu*self.sigma**4*(mn_mask_in[2:,1:-1] + mn_mask_in[:-2,1:-1] + mn_mask_in[1:-1,2:] + mn_mask_in[1:-1,:-2])/(self.dx*self.dy) + self.mu*self.sigma**2*self.s*(mn_mask_in[2:,1:-1] - mn_mask_in[:-2,1:-1])/(2*self.dy)
            self.m[mask_in] = S_gamma[mask_in[1:-1,1:-1]]/A_gamma[mask_in[1:-1,1:-1]]
            mn_mask_out = mn.copy()
            mn_mask_out[mask_inner_rim] = mn[mask_inner_rim]*self.u[mask_inner_rim]
            mn_x = (mn_mask_out[1:-1,2:] - mn_mask_out[1:-1, :-2])/(2*self.dx)
            mn_y = (mn_mask_out[2:,1:-1] - mn_mask_out[:-2,1:-1])/(2*self.dy)
            mn_xx = (mn_mask_out[2:,1:-1]+ mn_mask_out[:-2,1:-1] + mn_mask_out[1:-1, 2:]+ mn_mask_out[1:-1,:-2])/(self.dx*self.dy)
            A_m = 2*self.sigma**2/(self.dx*self.dy) - un_xx/self.mu
            S_m = 0.5*self.sigma**2*mn_xx + (un_x*mn_x + un_y*mn_y)/self.mu + self.s*mn_y
            self.m[1:-1,1:-1][mask_out[1:-1,1:-1]] = S_m[mask_out[1:-1,1:-1]]/A_m[mask_out[1:-1,1:-1]]
            l2norm = self.L2_error(self.m,mn)
            self.m =  self.alpha*self.m + (1-self.alpha)*mn
            if self.verbose and i%500 == 0:
                print('m error',l2norm)
            i+=1
            
    def jacobi(self):
        l2norm = 1
        while l2norm > self.l2_target:
            un = self.u.copy()
            A = -2*self.mu*self.sigma**4/(self.dx*self.dy) + self.lam + (self.g*self.m[1:-1,1:-1] + self.V[1:-1,1:-1])
            Q = un[1:-1,2:] + un[1:-1, :-2] + un[2:, 1:-1] + un[:-2, 1:-1]
            S = (-(0.5*self.mu*Q*self.sigma**4)/(self.dx*self.dy)+0.5*self.mu*(self.sigma**2)*self.s*(un[2:,1:-1] - un[:-2, 1:-1])/self.dy)
            self.u[1:-1,1:-1] = S/A
            l2norm = self.L2_error(self.u,un)
        
    def simulation(self,alpha = 'auto',save = False,verbose = False):
        """
        

        Parameters
        ----------
        alpha : float, optional
            Choose the mix parameter. The default is 'auto'.
        save : bool, optional
            If you want to save the data. The default is False.
        verbose : bool, optional
            If you want to show the progression. The default is False.

        Returns
        -------
        None.

        """
        
        if alpha != 'auto':
            self.alpha = alpha
        if self.mode == 'read':
            print('Cannot modify data in this mode. Try using write.')
        else: 
            if verbose:
                self.verbose = True
            tic  = time.time()
            l2norm = 1
            print('Computation begins')
            if self.gam > 0:
                while l2norm > self.l2_target:
                    m_old = self.m.copy()
                    self.jacobi_u()
                    self.jacobi_m()
                    self.m = self.alpha*self.m + (1-self.alpha)*m_old
                    l2norm = self.L2_error(self.m, m_old)
                    toc = time.time()
                    print(f'Error = {l2norm:.3e} Time = {(toc-tic)//3600:.0f}h{((toc-tic)//60)%60:.0f}m{(toc-tic)%60:.0f}s')
                print('Computation ends')
                mask_in =  np.sqrt(self.X**2 + self.Y**2) < (self.l + self.R)
                mask_out = np.sqrt(self.X**2 + self.Y**2) > (self.l + self.R)                
                p = self.u.copy()
                p[mask_out] = np.exp(-p[mask_out]/(self.mu*self.sigma**2))
                q = self.m.copy()
                q[mask_out] = q[mask_out]/p[mask_out]
                m = self.m
                self.m[mask_in] = m[mask_in]*self.u[mask_in]
            else:
                while l2norm > self.l2_target:
                    mn = self.m.copy()
                    self.jacobi()
                    p = self.u.copy()
                    q = np.flip(p,0)
                    self.m = self.alpha*p*q + (1-self.alpha)*mn
                    l2norm = self.L2_error(self.m,mn)
                    toc = time.time()
                    print(f'Error = {l2norm:.3e} Time = {(toc-tic)//3600:.0f}h{((toc-tic)//60)%60:.0f}m{(toc-tic)%60:.0f}s')
                print('Computation ends')
            self.get_velocities(p,q)    
            if save:
                np.savetxt(r'data/m_'+ self.config +'.txt',self.m)
                np.savetxt(r'data/vx_'+ self.config +'.txt',self.vx)
                np.savetxt(r'data/vy_'+ self.config +'.txt',self.vy)
    
    def draw_density(self,saturation = 'full', clim = 'auto',d = 'auto', title = True, colorbar = True, axis = True, scale = True, save = False,savedir = 'gfx'):   
        """
        Draw the density solution of the KFP equation

        Parameters
        ----------
        saturation : int, optional
            Changing this value changes the black-point of the colormap. The default is 'full'.
        clim : float, optional
            Max value of the colormap. The default is 'auto'.
        d : float, optional
            Size of the plot. The default is 'auto'.
        title : str, optional
            Title of the plot. The default is True.
        colorbar : bool, optional
            Choose if colorbar should appear. The default is True.
        axis : bool, optional
            Choose if axis should appear. The default is True.
        scale : bool, optional
            Choose if the scale on the bottom right should be shown. The default is True.
        save : bool, optional
            Choose if to save. The default is False.
        savedir : str, optional
            Choose where to save. The default is 'gfx'.

        Returns
        -------
        None.

        """
        # Enable LaTeX
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        # Create colormap
        usual = mpl.cm.hot_r(np.arange(256))
        if saturation == 'full':
            saturation = 256
        saturate = np.ones((int(256/saturation),4))
        for i in range(3):
            saturate[:,i] = np.linspace(usual[-1,i],0,saturate.shape[0])
        cmap = np.vstack((usual,saturate))
        cmap = mpl.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])
        plt.figure(figsize=(10,10))
        if d == 'auto':
            d = min(self.lx,self.ly)
        plt.xlim((-d,d))
        plt.ylim((-d,d))
        if title:
            plt.title(fr'$s$={self.s:.2f} $c_s$={self.c_s:.2f} $R$={self.R:.2f} $\xi$={self.xi:.2f} $\gamma$={self.gam:.2f} $m_0$={self.m_0:.1f}',size= 20)
        plt.xticks([-d,0,d],[-d,0,d],size = 20)
        plt.yticks([-d,0,d],[-d,0,d],size = 20)
        if scale: 
            scale = plt.arrow(d - 1.2, -d + 0.2, 1, 0,width = .1,head_width = 0,head_length = 0,color = 'lime',zorder= 10)
            plt.text(d - 0.7, -d + 0.4, r'1m',size = 20, color = 'lime', ha = 'center', va = 'center')
            plt.gca().add_artist(scale)
        if not axis:
            plt.axis('off')
        a = plt.arrow(0,-0.2*(self.R/.37),0,0.25*(self.R/.37),width = .1*(self.R/.37),head_width = .3*(self.R/.37),head_length = .2*(self.R/.37),color = 'black',zorder= 10)
        c = plt.Circle((0, 0),radius = self.R)
        plt.gca().add_artist(a)
        plt.gca().add_artist(c)
        plt.imshow(np.flip(self.m,axis = 0),extent=[-self.lx,self.lx,-self.ly,self.ly],cmap = cmap)
        if clim != 'auto':
            plt.clim(0,clim)
        if colorbar:
            plt.colorbar()
        if save:
            plt.savefig(r''+ savedir+'/'+self.config+'.png',bbox_inches='tight', pad_inches=0)
    
    def draw_velocities(self, l = 'auto', d = 'auto', title = True, colorbar = True, scale = True, axis = True, save = False,savedir = 'gfx'):
        """
        Draw the density solution of the KFP equation

        Parameters
        ----------
        l : TYPE, optional
            DESCRIPTION. The default is 'auto'.
        d : float, optional
            Size of the plot. The default is 'auto'.
        title : str, optional
            Title of the plot. The default is True.
        colorbar : bool, optional
            Choose if colorbar should appear. The default is True.
        axis : bool, optional
            Choose if axis should appear. The default is True.
        scale : bool, optional
            Choose if the scale on the bottom right should be shown. The default is True.
        save : bool, optional
            Choose if to save. The default is False.
        savedir : str, optional
            Choose where to save. The default is 'gfx'.

        Returns
        -------
        None.

        """
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.figure(figsize=(10,10))
        if d == 'auto':
            d = min(self.lx,self.ly)
        if l =='auto':
            lx = int(round((self.nx - 2)/40))
            ly = int(round((self.ny - 2)/40))
            l = np.min([lx,ly])
        plt.xlim((-d,d))
        plt.ylim((-d,d))
        if title:
            plt.title(fr'$s$={self.s:.2f} $c_s$={self.c_s:.2f} $R$={self.R:.2f} $\xi$={self.xi:.2f} $\gamma$={self.gam:.2f} $m_0$={self.m_0:.1f}',size= 20)
        plt.xticks([-d,0,d],[-d,0,d],size = 20)
        plt.yticks([-d,0,d],[-d,0,d],size = 20)
        if not axis:
            plt.axis('off')
        a = plt.arrow(0,-0.2*(self.R/.37),0,0.25*(self.R/.37),width = .1*(self.R/.37),head_width = .3*(self.R/.37),head_length = .2*(self.R/.37),color = 'black',zorder= 10)
        c = plt.Circle((0, 0),radius = self.R)
        plt.gca().add_artist(a)
        plt.gca().add_artist(c)
        x = self.X[1:-1,1:-1][::l,::l]
        y = self.Y[1:-1,1:-1][::l,::l]
        mtr = self.m[1:-1,1:-1]
        mtr = mtr[::l,::l]
        mtr = (mtr-np.min(mtr))/np.max(mtr)
        ax = self.vx[::l,::l]
        ay = self.vy[::l,::l]
        plt.quiver(x,y,ax,ay + self.s,angles='xy', scale_units='xy', scale=1, pivot = 'mid', alpha = mtr)
        if scale: 
            scale = plt.arrow(d - 1.2, -d + 0.2, 1, 0, width = .08,head_width = 0.15,head_length = 0.15,color = 'red',zorder= 10)
            plt.text(d - 0.7, -d + 0.4, r'1m',size = 20, color = 'red', ha = 'center', va = 'center')
            plt.gca().add_artist(scale)
        if save:
            plt.savefig(r''+ savedir+'/'+self.config+'.png')
        plt.show()
        plt.close()
        
    def save(self):
        np.savetxt(r'data/m_'+ self.config +'.txt',self.m)
        np.savetxt(r'data/vx_'+ self.config +'.txt',self.vx)
        np.savetxt(r'data/vy_'+ self.config +'.txt',self.vy)
        
    def get_velocities(self,p,q):
        dx = self.dx
        dy = self.dy
        phi_grad_x = (p[1:-1,2:]-p[1:-1,:-2])/(2*dx)
        phi_grad_y = (p[2:,1:-1]-p[:-2,1:-1])/(2*dy)
        gamma_grad_x = (q[1:-1,2:]-q[1:-1,:-2])/(2*dx)
        gamma_grad_y = (q[2:,1:-1]-q[:-2,1:-1])/(2*dy)
        self.vx = self.sigma**2*(q[1:-1,1:-1]*phi_grad_x-p[1:-1,1:-1]*gamma_grad_x)/(2*self.m[1:-1,1:-1])
        self.vy = self.sigma**2*(q[1:-1,1:-1]*phi_grad_y-p[1:-1,1:-1]*gamma_grad_y)/(2*self.m[1:-1,1:-1]) - self.s           