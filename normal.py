# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 12:53:47 2019

@author: Administrator
"""

from PIL import Image
import os
import numpy as np
import cv2


def all_pics_processing(root , transformed_images):
    for path,dirs,files in os.walk(root):
        break
    
    iiii = 0
    for pic_name in files:
        iiii += 1
        if iiii % 100 == 0:
            print('正在处理第'+str(iiii) + "张图片。")
        pathss=os.path.join(path,pic_name)
        image=Image.open(pathss)
        image=image.resize((72,72))
        img = np.asarray(image,dtype = 'float32')
        shape = img.shape
            
        output_file = os.path.join(transformed_images,pic_name)
            
        if not os.path.exists(transformed_images):
            os.makedirs(transformed_images)
                
                
        if len(shape) == 2:
            image = cv2.imread(pathss)
            img = cv2.cvtColor(cv2.resize(image,(72,72)),cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_file,img)
            
        elif len(shape) == 3:
            image.save(output_file)
                
            
                
            
            
            
if __name__ == '__main__':
    root = './train/'
    transformed_images = './new_train/'
    all_pics_processing(root,transformed_images)