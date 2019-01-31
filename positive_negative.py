# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 20:54:13 2019

@author: Administrator
"""
import csv_deal
import random

file_name = 'new_train.csv'

def generation():
    
    
    f1 = open("./datas/dataset_id_num.txt",'r')
    f2 = open("./datas/dataset_path.txt",'r')
    f3 = open("./datas/positive_pairs_path.txt",'w')
    f4 = open("./datas/negative_pairs_path.txt",'w')
    
    
    nums_line = f1.readline()
    id_nums = nums_line.strip().split(' ')
    paths_line = f2.readline()
    paths = paths_line.strip().split(' ')
    
    transform_now = 0
    for id_num in id_nums:
        id_nums[transform_now] = eval(id_num)
        

    
    '''-------------------以下是positive生成程序---------------------------'''
    
    '''
    此处，判断单一图片出现次数，
    以start为起点加上出现次数，
    在此范围内进行无顺序自由组合
    '''
    
    start = 0
    idnum = id_nums[0]
    index = 1
    for id_num in id_nums[1:]:
        
        '''
        此处计数，当和前一个的id相同时
        index加一，当不同时index置1，
        并用times来记录出现多少次
        '''
        
        if idnum == id_num:
            index += 1
        else:
            idnum = id_num
            times = index
            index = 1


        '''
        此时选用二重循环，允许出现自己和自己成对的情况，
        因为有的图片只有一张，他们是和自己训练的，
        为了各个样本公平训练，其它也有自己和自己成对的机会
        最后id为new_whale，并非某一条鲸鱼，
        而是所有未编号的鲸鱼的集合，依然把他们视为同一条鲸鱼，
        目的是训练出一片不识别的大区域，后续的未出现在数据集中的
        就会出现在new_whale 范围中，归类为new_whale
        '''    
        
        if index == 1:
            for i in range(times):
                for j in range(i,times):
                    if(i != j):
                        f3.write(str(paths[start+i]) + ' ' + str(paths[start+j]) + '\n')
            start += times
        
    
    
    '''-------------------以下是negative生成程序---------------------------'''
    
    '''
    和自己id不同的配对,和每条鲸鱼的图片仅配对一张图片
    '''    
    now = 0
    random_index = 0
    for id_num in id_nums:
        index = now
        flag = True
        
        while(index < 298243):
            iid = id_nums[index]
            if (iid != id_nums[index - random_index]):
                flag = True
                
            if (id_num != id_nums[index]) & flag :
                f4.write(str(paths[now]) + ' ' + str(paths[index]) + '\n')
                flag = False
            random_index = random.randint(1500,2000)
            index += random_index
        now += 1
        
        if now % 100 ==0:
            print("现在正在处理第"+str(now)+"个样本的负样本")
                    
    f1.close()
    f2.close()
    f3.close()
    f4.close()        


def main():
    csv_deal.csv_deal(file_name)
    generation()
    
if __name__ == '__main__' :
    main()