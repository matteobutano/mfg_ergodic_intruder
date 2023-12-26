import os
from send2trash import send2trash

path = 'data'

configs = [config[:-5] for config in os.listdir('configs')]

for datas in os.listdir(path):
    
    name = datas.split('_',1)[1][:-4]

    if name not in configs:
        
        send2trash(path+'/'+datas)
