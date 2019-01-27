# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 12:53:47 2019

@author: Administrator
"""

from PIL import Image
import os
import xlrd
import numpy as np
import cv2
import csv


def all_pics_processing(root , transformed_images,enhancement = True):
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
            
        
            
        if not os.path.exists(transformed_images):
            os.makedirs(transformed_images)
                
        if (enhancement):     
            if len(shape) == 2:
                image = cv2.imread(pathss)
                img = cv2.cvtColor(cv2.resize(image,(72,72)),cv2.COLOR_BGR2RGB)
                img = np.asarray(img,np.uint8)
                image = Image.fromarray(img)
                
                
                
            #生成原样本的图片    
            new_pic_name = '0_'+ str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image.save(output_file)
            
            #生成水平翻转图片
            new_pic_name = '1_'+ str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_fan = image.transpose(Image.FLIP_LEFT_RIGHT)
            image_fan.save(output_file)
            
            #生成逆时针旋转30度图片
            image_30 = image.rotate(30)
            new_pic_name = '2_'+ str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_30.save(output_file)
            
            #生成逆时针旋转60度的图片
            image_60 = image.rotate(60)
            new_pic_name = '3_'+ str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_60.save(output_file)
            
            #生成顺时针旋转60度的图片
            image_300 = image.rotate(300)
            new_pic_name = '4_'+ str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_300.save(output_file)
            
            #生成顺时针旋转30度的图片
            image_330 = image.rotate(330)
            new_pic_name = '5_'+ str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_330.save(output_file)
            
            #生成噪点图片
            image_with_noise = get_noise(image)
            new_pic_name = '6_'+str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_with_noise.save(output_file)
            
            #生成带噪点的水平翻转图片
            new_pic_name = '7_'+ str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_fan = image_with_noise.transpose(Image.FLIP_LEFT_RIGHT)
            image_fan.save(output_file)
            
            #生成带噪点的水平逆时针旋转30度照片
            image_30 = image_with_noise.rotate(30)
            new_pic_name = '8_'+ str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_30.save(output_file)
            
            #生成带噪点的逆时针旋转30度的照片
            image_60 = image_with_noise.rotate(60)
            new_pic_name = '9_'+ str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_60.save(output_file)
            
            #生成带噪点的顺时针旋转60度的照片
            image_300 = image_with_noise.rotate(300)
            new_pic_name = '10_'+ str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_300.save(output_file)
            
            #生成带噪点的顺时针旋转30度的照片
            image_330 = image_with_noise.rotate(330)
            new_pic_name = '11_'+ str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_330.save(output_file)
            
            #缩放过后的图片，上往下减去十个像素行，并重新resize到72*72
            image_top_crop = image.crop((0,10,72,72))
            image_top_crop = image_top_crop.resize((72,72))
            new_pic_name = '12_'+ str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_top_crop.save(output_file)
            
            #缩放过后的图片，下往上减去十个像素行，并重新resize到72*72
            image_bottom_crop = image.crop((0,0,72,62))
            image_bottom_crop = image_bottom_crop.resize((72,72))
            new_pic_name = '13_'+ str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_bottom_crop.save(output_file)
            
            #缩放过后加入噪点的图片，上往下减去十个像素行，并重新resize到72*72
            image_top_crop_with_noise = image_with_noise.crop((0,10,72,72))
            image_top_crop_with_noise = image_top_crop_with_noise.resize((72,72))
            new_pic_name = '14_'+ str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_top_crop_with_noise.save(output_file)
            
            #缩放过后加入噪点的图片，下往上减去十个像素行，并重新resize到72*72
            image_bottom_crop_with_noise = image_with_noise.crop((0,0,72,62))
            image_bottom_crop_with_noise = image_bottom_crop_with_noise.resize((72,72))
            new_pic_name = '15_'+ str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_bottom_crop_with_noise.save(output_file)
          
        else:
            if len(shape) == 2:
                image = cv2.imread(pathss)
                img = cv2.cvtColor(cv2.resize(image,(72,72)),cv2.COLOR_BGR2RGB)
                img = np.asarray(img,dtype = 'float32')
                image = Image.fromarray(img)            
        
            output_file = os.path.join(transformed_images,pic_name)
            image.save(output_file)



'''此函数是获得噪声的函数'''
def get_noise(img):
    img = np.asarray(img,np.uint8)
    img.flags.writeable = True
    row_num = 0
    for row in img:
        for i in range(1):
            img[row_num,np.random.randint(0,72),:] = 255
        row_num += 1
        
    img = Image.fromarray(img)          
        
    return img                                  #注意：此处返回去的是Image类型的文件
                
def csv_enhancement():
    workbook = xlrd.open_workbook('train.csv')     #从按图片排序的csv中读取
    imginfo = workbook.sheet_by_index(0)
    paths = imginfo.col_values(0)
    idds = imginfo.col_values(1)
    
    f1 = open('new_train.csv','w',newline='')
    writer = csv.writer(f1)
    tt = 0
    
    for path in paths:
        path_num = 0
        for i in range(16):
            path_name = str(path_num).strip()+'_'+str(path).strip()
            writer.writerow((path_name,idds[tt]))
            path_num += 1
        tt += 1

            
            
            
if __name__ == '__main__':
    root = './train/'
    transformed_images = './new_train/'
    all_pics_processing(root,transformed_images)
    csv_enhancement()
    