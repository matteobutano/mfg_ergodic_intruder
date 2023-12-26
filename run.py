import mfg_ergodic_intruder.mfg_ergodic as mfg
import matplotlib.pyplot as plt

m = mfg.mfg('top_left_small', 'write')

m.simulation(verbose = True)