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
    print(len(files))
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
            
            
            #生成噪点图片
            image_with_noise = get_noise(image)
            new_pic_name = '2_'+str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_with_noise.save(output_file)
            
            #生成带噪点的水平翻转图片
            new_pic_name = '3_'+ str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_fan = get_noise(image.transpose(Image.FLIP_LEFT_RIGHT))
            image_fan.save(output_file)
            
            
            #随机生成3组0~90
            for i in range(4,6):
                degree = np.random.randint(0,60)
                new_pic_name = str(i) + '_' + str(pic_name).strip()
                output_file = os.path.join(transformed_images,new_pic_name)
                image_new = image.rotate(degree)
                image_new.save(output_file)
                
                
            #生成0~-90度的随机旋转图片    
            for i in range(6,10):
                degree = np.random.randint(300,359)
                new_pic_name = str(i) + '_' + str(pic_name).strip()
                output_file = os.path.join(transformed_images,new_pic_name)
                image_new = image.rotate(degree)
                image_new.save(output_file)
            
            #生成0~90度的带噪点随机旋转图片
            for i in range(10,13):
                degree = np.random.randint(0,60)
                new_pic_name = str(i) + '_' + str(pic_name).strip()
                output_file = os.path.join(transformed_images,new_pic_name)
                noise_level = np.random.randint(1,4)
                image_new = get_noise(image.rotate(degree),noise_level)
                image_new.save(output_file)
            
            #生成0~-90度带噪点的随机角度旋转图片
            for i in range(13,16):
                degree = np.random.randint(300,359)
                new_pic_name = str(i) + '_' + str(pic_name).strip()
                output_file = os.path.join(transformed_images,new_pic_name)
                noise_level = np.random.randint(1,3)
                image_new = get_noise(image.rotate(degree),noise_level)
                image_new.save(output_file)
            
            #生成3组不同范围的剪切图片    
            for i in range(16,19):
                pixel_need_to_cut_left = np.random.randint(8,20)
                pixel_need_to_cut_right = np.random.randint(8,20)
                pixel_need_to_cut_top = np.random.randint(10,20)
                pixel_need_to_cut_bottom = np.random.randint(10,20)
                image_new = image.crop((pixel_need_to_cut_left,pixel_need_to_cut_top,72 - pixel_need_to_cut_right,72 - pixel_need_to_cut_bottom))
                image_new = image_new.resize((72,72))
                new_pic_name = str(i) + '_' + str(pic_name).strip()
                output_file = os.path.join(transformed_images,new_pic_name)
                image_new.save(output_file)
          
            '''生成三组剪切一半图片,
                cut_direction == 0则减去左半边，
                为1则减去右边半边
                为2则减去上半边，
                为3则减去下半边
            '''
            for i in range(19,22):
                cut_direction = np.random.randint(4)
                if cut_direction == 0:
                    pixel_need_to_cut_left = np.random.randint(20,40)
                    pixel_need_to_cut_right = np.random.randint(0,5)
                    pixel_need_to_cut_top = np.random.randint(1,10)
                    pixel_need_to_cut_bottom = np.random.randint(1,10)
                    image_new = image.crop((pixel_need_to_cut_left,pixel_need_to_cut_top,72 - pixel_need_to_cut_right,72 - pixel_need_to_cut_bottom))
                    image_new = image_new.resize((72,72))
                    new_pic_name = str(i) + '_' + str(pic_name).strip()
                    output_file = os.path.join(transformed_images,new_pic_name)
                    image_new.save(output_file)
                elif cut_direction == 1:
                    pixel_need_to_cut_left = np.random.randint(0,5)
                    pixel_need_to_cut_right = np.random.randint(20,40)
                    pixel_need_to_cut_top = np.random.randint(1,10)
                    pixel_need_to_cut_bottom = np.random.randint(1,10)
                    image_new = image.crop((pixel_need_to_cut_left,pixel_need_to_cut_top,72 - pixel_need_to_cut_right,72 - pixel_need_to_cut_bottom))
                    image_new = image_new.resize((72,72))
                    new_pic_name = str(i) + '_' + str(pic_name).strip()
                    output_file = os.path.join(transformed_images,new_pic_name)
                    image_new.save(output_file)
                elif cut_direction == 2:
                    pixel_need_to_cut_left = np.random.randint(0,10)
                    pixel_need_to_cut_right = np.random.randint(1,10)
                    pixel_need_to_cut_top = np.random.randint(20,40)
                    pixel_need_to_cut_bottom = np.random.randint(1,5)
                    image_new = image.crop((pixel_need_to_cut_left,pixel_need_to_cut_top,72 - pixel_need_to_cut_right,72 - pixel_need_to_cut_bottom))
                    image_new = image_new.resize((72,72))
                    new_pic_name = str(i) + '_' + str(pic_name).strip()
                    output_file = os.path.join(transformed_images,new_pic_name)
                    image_new.save(output_file)
                elif cut_direction == 3:
                    pixel_need_to_cut_left = np.random.randint(0,10)
                    pixel_need_to_cut_right = np.random.randint(1,10)
                    pixel_need_to_cut_top = np.random.randint(1,5)
                    pixel_need_to_cut_bottom = np.random.randint(20,40)
                    image_new = image.crop((pixel_need_to_cut_left,pixel_need_to_cut_top,72 - pixel_need_to_cut_right,72 - pixel_need_to_cut_bottom))
                    image_new = image_new.resize((72,72))
                    new_pic_name = str(i) + '_' + str(pic_name).strip()
                    output_file = os.path.join(transformed_images,new_pic_name)
                    image_new.save(output_file)
        else:
            if len(shape) == 2:
                image = cv2.imread(pathss)
                img = cv2.cvtColor(cv2.resize(image,(72,72)),cv2.COLOR_BGR2RGB)
                img = np.asarray(img,dtype = 'float32')
                image = Image.fromarray(img)            
        
            output_file = os.path.join(transformed_images,pic_name)
            image.save(output_file)



'''此函数是获得噪声的函数'''
def get_noise(img,noise_level = 1):
    img = np.asarray(img,np.uint8)
    img.flags.writeable = True
    row_num = 0
    for row in img:
        for i in range(noise_level):
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
        for i in range(22):
            path_name = str(path_num).strip()+'_'+str(path).strip()
            writer.writerow((path_name,idds[tt]))
            path_num += 1
        tt += 1

            
            
            
if __name__ == '__main__':
    root = './train/'
    transformed_images = './new_train/'
    all_pics_processing(root,transformed_images)
    csv_enhancement()
    