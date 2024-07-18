# Solution of a Mean-Field Game's Stationary State with Discount Factor
This software allows for the solution of the stationary state of the Mean-Field Games model described in publications [Bonnemain et al.](https://arxiv.org/abs/2201.08592) and [Butano et al.](https://arxiv.org/abs/2302.08945), modelling the passage through a dense crowd of a cylindrical obstacle at constant speed $s$. The equations we want to solve are 

$$0 =\frac{\sigma^2}{2}\Delta{u^e} - \frac{1}{2\mu}(\vec{\nabla}u^e)^2  - \gamma u^e(x) - V[m^e]$$

$$0 =\frac{\sigma^2}{2}\Delta m^e  +\frac{1}{\mu}\nabla\cdot(m^e\nabla u^e)$$

We do this by using a double formulation, using the Cole-Hopf change of variable allowing an elegant connection to the theory of the Non-Linear Schrödinger Equation. 

## Install and Use the Module

To use this python module, first of all create a directory named "project_mfg". Inside this folder, create the following directories
- data
- gfx
- configs

Then, create the python script "run.py", which contains the instructions
- *import mfg_ergodic_intruder.mfg_ergodic as mfg* : to import the module. 
- *m = mfg.mfg('config_name', 'mode')* : where 'config_name' is the name of a configuration that should be formatted like [this](https://github.com/user-attachments/files/16283880/empty_config.json). 
- *m.simulation()* : to launch the simulation. Options in the docu. 
- *m.draw_density()* : to draw the MFG density. Options in the docu. 
- *m.draw_velocities()* : to draw the MFG velocities. Options in the docu. 
