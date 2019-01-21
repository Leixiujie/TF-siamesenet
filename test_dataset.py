import numpy as np
from PIL import Image
from config import FLAGS
import os

base_path = './datas/test_pair_lists/'
for path,dirs,files in os.walk(base_path):
        break

for txt in files:
    txt_path = os.path.join(base_path,str(txt).strip())
    comparison_file = open(txt_path, 'r')
    comparison_pairs_path_line = comparison_file.readlines()
    
    for line in comparison_pairs_path_line:
        left_right = line.strip().split(' ')
        
    

    