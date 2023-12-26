import os
from send2trash import send2trash

data = [file[2:-4] for file in os.listdir('data') if file[0]=='m']

for file in os.listdir('outputs'):
    if file[:-4] in data:
        send2trash('outputs/'+file)

print('Outputs cleaned')
