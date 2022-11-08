import os
import shutil
import argparse

print("few shot manager")
os.system('python main.py  -t -time -part  --batch 50' )
os.system('python main.py  -t -time -part -overclass -union   --Kuniq 0.05 --Kunif  1 --batch 50 ' )
os.system('python main.py  -t -time -part  -union    --batch 50 ' )



# lst = os.listdir('3overclass exp')
# for address in lst:
#     cmd = 'python main.py  -r  -overclass  --extraclass 3 -openset  --Kuniq 1 --Kunif  3 --load_path "3overclass exp/' + address +'"'
#     os.system(cmd)