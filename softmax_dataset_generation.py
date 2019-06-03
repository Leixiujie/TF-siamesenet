# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 12:53:47 2019

@author: Administrator
"""

from PIL import Image
import os
import xlrd
import csv
import numpy as np
import cv2
import csv
import tf_enhancement


def all_pics_processing(root , transformed_images,enhancement = True):
    for path,dirs,files in os.walk(root):
        break
    print(len(files))
    iiii = 0
    for pic_name in files:
        iiii += 1
        if iiii % 10.0 == 0:
            print('正在处理第'+str(iiii) + "张图片。")
        pathss=os.path.join(path,pic_name)
        image=Image.open(pathss)
        image=image.resize((600,600))                           #不用tf处理的时候用
        img = np.asarray(image,dtype = 'float32')
        shape = img.shape
            
            
        if not os.path.exists(transformed_images):
            os.makedirs(transformed_images)
                
        if (enhancement):     
            if len(shape) == 2:
                image = cv2.imread(pathss)
                img = cv2.cvtColor(cv2.resize(image,(600,600)),cv2.COLOR_BGR2RGB)
                img = np.asarray(img,np.uint8)
                image = Image.fromarray(img)
                
                
                
            #生成原样本的图片    
            new_pic_name = '0_'+ str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_=image.resize((224,224)) 
            image_.save(output_file)
            
            #生成水平翻转图片
            new_pic_name = '1_'+ str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_fan = (image.transpose(Image.FLIP_LEFT_RIGHT)).resize((224,224)) 
            image_fan.save(output_file)
            
            
            #生成噪点图片
            image_with_noise = (get_noise(image)).resize((224,224)) 
            new_pic_name = '2_'+str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_with_noise.save(output_file)
            
            #生成带噪点的水平翻转图片
            new_pic_name = '3_'+ str(pic_name).strip()
            output_file = os.path.join(transformed_images,new_pic_name)
            image_fan = (get_noise(image.transpose(Image.FLIP_LEFT_RIGHT))).resize((224,224)) 
            image_fan.save(output_file)
            
            
            #随机生成10组0~60带噪点的旋转图片
            for i in range(4,6):
                degree = np.random.randint(0,60)
                new_pic_name = str(i) + '_' + str(pic_name).strip()
                output_file = os.path.join(transformed_images,new_pic_name)
                noise_level = np.random.randint(1,3)
                image_new = (get_noise(image.rotate(degree),noise_level)).resize((224,224)) 
                image_new.save(output_file)
                
                
            #生成0~-60度带噪点的随机角度旋转图片
            for i in range(6,8):
                degree = np.random.randint(300,359)
                new_pic_name = str(i) + '_' + str(pic_name).strip()
                output_file = os.path.join(transformed_images,new_pic_name)
                noise_level = np.random.randint(1,3)
                image_new = (get_noise(image.rotate(degree),noise_level)).resize((224,224)) 
                image_new.save(output_file)
            '''
            #生成10组不同范围的剪切图片    
            for i in range(10,12):
                pixel_need_to_cut_left = np.random.randint(0,0.3*600)
                pixel_need_to_cut_right = np.random.randint(0,0.3*600)
                pixel_need_to_cut_top = np.random.randint(0,0.3*600)
                pixel_need_to_cut_bottom = np.random.randint(0,0.3*600)
                degree = np.random.randint(-60,60)
                image_new = (image.rotate(degree)).crop((pixel_need_to_cut_left,pixel_need_to_cut_top,600 - pixel_need_to_cut_right,600 - pixel_need_to_cut_bottom))
                image_new = image_new.resize((224,224))
                new_pic_name = str(i) + '_' + str(pic_name).strip()
                output_file = os.path.join(transformed_images,new_pic_name)
                image_new.save(output_file)
            '''
            '''生成三组剪切一半图片,
                cut_direction == 0则减去左半边，
                为1则减去右边半边
                为2则减去上半边，
                为3则减去下半边
            '''
            for i in range(8,11):
                cut_direction = 0
                if cut_direction == 0:
                    pixel_need_to_cut_left = np.random.randint(int(35.0/72.0*600),int(50.0/72.0*600))
                    pixel_need_to_cut_right = np.random.randint(0,90)
                    pixel_need_to_cut_top = np.random.randint(1,90)
                    pixel_need_to_cut_bottom = np.random.randint(1,90)
                    image_new = image.crop((pixel_need_to_cut_left,pixel_need_to_cut_top,600 - pixel_need_to_cut_right,600 - pixel_need_to_cut_bottom))
                    image_new = image_new.resize((224,224))
                    new_pic_name = str(i) + '_' + str(pic_name).strip()
                    output_file = os.path.join(transformed_images,new_pic_name)
                    image_new.save(output_file)
                elif cut_direction == 1:
                    pixel_need_to_cut_left = np.random.randint(0,90)
                    pixel_need_to_cut_right = np.random.randint(int(35.0/72.0*600),int(50.0/72.0*600))
                    pixel_need_to_cut_top = np.random.randint(1,90)
                    pixel_need_to_cut_bottom = np.random.randint(1,90)
                    image_new = image.crop((pixel_need_to_cut_left,pixel_need_to_cut_top,600 - pixel_need_to_cut_right,600 - pixel_need_to_cut_bottom))
                    image_new = image_new.resize((224,224))
                    new_pic_name = str(i) + '_' + str(pic_name).strip()
                    output_file = os.path.join(transformed_images,new_pic_name)
                    image_new.save(output_file)
                elif cut_direction == 2:
                    pixel_need_to_cut_left = np.random.randint(0,90)
                    pixel_need_to_cut_right = np.random.randint(1,90)
                    pixel_need_to_cut_top = np.random.randint(int(20.0/72.0*600),int(35.0/72.0*600))
                    pixel_need_to_cut_bottom = np.random.randint(1,90)
                    image_new = image.crop((pixel_need_to_cut_left,pixel_need_to_cut_top,600 - pixel_need_to_cut_right,600 - pixel_need_to_cut_bottom))
                    image_new = image_new.resize((224,224))
                    new_pic_name = str(i) + '_' + str(pic_name).strip()
                    output_file = os.path.join(transformed_images,new_pic_name)
                    image_new.save(output_file)
                elif cut_direction == 3:
                    pixel_need_to_cut_left = np.random.randint(0,90)
                    pixel_need_to_cut_right = np.random.randint(1,90)
                    pixel_need_to_cut_top = np.random.randint(1,90)
                    pixel_need_to_cut_bottom = np.random.randint(int(20.0/72.0*600),int(35.0/72.0*600))
                    image_new = image.crop((pixel_need_to_cut_left,pixel_need_to_cut_top,600 - pixel_need_to_cut_right,600 - pixel_need_to_cut_bottom))
                    image_new = image_new.resize((224,224))
                    new_pic_name = str(i) + '_' + str(pic_name).strip()
                    output_file = os.path.join(transformed_images,new_pic_name)
                    image_new.save(output_file)
                cut_direction += 1
        
        else:
            if len(shape) == 2:
                image = cv2.imread(pathss)
                img = cv2.cvtColor(cv2.resize(image,(600,600)),cv2.COLOR_BGR2RGB)    #不用tf处理的时候用
                img = np.asarray(img,dtype = np.uint8)
                image = Image.fromarray(img)            
        
            output_file = os.path.join(transformed_images,pic_name)
            image = image.resize((224,224))
            image.save(output_file)


'''此函数是获得噪声的函数'''
def get_noise(img,noise_level = 1):
    noise_level = 10
    img = np.asarray(img,np.uint8)
    img.flags.writeable = True
    row_num = 0
    for row in img:
        for i in range(noise_level):
            img[row_num,np.random.randint(0,600),:] = 255
        row_num += 1
        
    img = Image.fromarray(img)          
        
    return img                                  #注意：此处返回去的是Image类型的文件
                
def csv_enhancement():
    '''
    xlrd版本加强csv
    workbook = xlrd.open_workbook('train.csv')     #从按图片排序的csv中读取
    imginfo = workbook.sheet_by_index(0)
    paths = imginfo.col_values(0)
    idds = imginfo.col_values(1)
    
    f1 = open('softmax_dataset.csv','w',newline='')
    f2 = open('./datas/softmax_path.txt','w')
    writer = csv.writer(f1)
    tt = 0
    

    for path in paths:
        

        for i in range(0,64):
            path_name = str(i).strip()+'_'+str(path).strip()
            writer.writerow((path_name,idds[tt]))
        
        
                
        tt += 1
    
    
    f1.close()
    f2.close()
    '''
    with open('train.csv') as f:
        reader = csv.reader(f)
        f1 = open('softmax_dataset.csv','w',newline='')
        writer = csv.writer(f1)
        iii = 0
        for row in reader:
            if iii == 15697:
                break
            for i in range(0,64):
                path_name = str(i).strip()+'_'+str(row[0]).strip()
                writer.writerow((path_name,row[1]))
            iii += 1
        f1.close()
            
if __name__ == '__main__':
    root = './train/'
    transformed_images = './softmax_train/'
    #csv_enhancement()
    all_pics_processing(root,transformed_images)
    all_pics_processing('./train/','./train_with_3_channels/',enhancement=False)
    tf_enhancement.generate('./train_with_3_channels/','./softmax_train/')
    #all_pics_processing('./test/','./new_test/',enhancement=False)    
