import os
import sys
from send2trash import send2trash

# file = sys.argv[1]

file = 'bottom_right_intermediate'

send2trash('data/m_'+file+'.txt')
send2trash('data/vx_'+file+'.txt')
send2trash('data/vy_'+file+'.txt')


