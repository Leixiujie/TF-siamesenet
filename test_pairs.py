import normal
import os
import tensorflow as tf
from config import FLAGS

root = './test/'
transformed_images = './new_test/'
transformed_train_images = './new_train/'

def test_pic_processing():
    normal.all_pics_processing(root , transformed_images)
    print('--------图片处理完成-------------')
    
def test_pairs_generation(test_images_path):
    lists_path = './datas/test_pair_lists/'
    if not os.path.exists(lists_path):
        os.makedirs(lists_path)
    
    for path,dirs,files in os.walk(test_images_path):
        break
    
    f = open('./datas/dataset_path.txt','r')
    f1 = open("./datas/dataset_id_num.txt",'r')
    paths = f.readline()
    id_nums = f1.readline()
    paths = paths.strip().split(' ')
    id_nums = id_nums.strip().split(' ')
    '''
    此处将每一个test里面的图片，列出一个与训练集里图片的配对表
    （全都放在一个txt里面文件太大，大概5.7g，
    转换为图片矩阵再载入内存不太够，此方法较为暴力....
    但是对于竞赛估计有效，毕竟只呈交一个csv文件）
    '''
    iii = 0
    for file in files:
        iii += 1
        if(iii % 50 == 0):
            print('已生成' + str(iii) + '/7960个图片对')
        pic_name = str(file).strip().split('.')[0]
        txt_name = str(pic_name) + '.txt'
        total_name = os.path.join(lists_path,txt_name)
        f = open(total_name,'w')
        '''
        nuum = 0
        previous = 0
        for path in paths[:15697]:
            if(id_nums[nuum] != previous):
                f.write('./new_train/' + str(path) + ' ' + './new_test/' + str(file) + '\n')
                previous = id_nums[nuum]
            nuum += 1
       '''     
        for path in paths:
            f.write('./new_train/' + str(path) + ' ' + './new_test/' + str(file) + '\n')
        f.close()
        f1.close()

    

if __name__ == '__main__':
    #test_pic_processing()
    test_pairs_generation(transformed_images)