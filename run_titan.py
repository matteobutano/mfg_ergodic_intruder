from mfg_ergodic_intruder import mfg_ergodic
import sys

config = sys.argv[1]

m = mfg_ergodic.mfg(config,'write')

m.simulation(0.2,verbose = True, save=True)
