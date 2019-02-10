import numpy as np
import tensorflow as tf
from PIL import Image
from config import FLAGS
import os

DEV_NUMBER = FLAGS.DEV_NUMBER
BASE_PATH = './softmax_train/'
batch_size = FLAGS.batch_size

if not os.path.exists(FLAGS.negative_file):
        print('------------------文件不存在---------------------')
        

softmax_path_file = open(FLAGS.softmax_dataset, 'r')
softmax_pic_paths = softmax_path_file.readlines()
print(len(softmax_pic_paths))

print("mark: loaded softmax dataset's paths")
softmax_paths = []
id_num = []

for line in softmax_pic_paths:
    path_and_id_num = line.strip().split(' ')
    softmax_paths.append(path_and_id_num[0])
    id_num.append(eval(path_and_id_num[1]))
softmax_pic_paths = []
print('mark:softmax paths loaded')


print('mark: added positve and negative array ')
softmax_paths = np.asarray(softmax_paths)
id_num = np.asarray(id_num)


np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(softmax_paths)))
id_num_shuffled = id_num[shuffle_indices]
id_num = []

softmax_paths_shuffled = softmax_paths[shuffle_indices]
softmax_paths = []


#生成测试和训练集合
train_paths = softmax_paths_shuffled[:DEV_NUMBER]
dev_paths = softmax_paths_shuffled[DEV_NUMBER:]
softmax_paths_shuffled = []

train_id_num, dev_id_num = id_num_shuffled[:DEV_NUMBER], id_num_shuffled[DEV_NUMBER:]
id_num_shuffled = []
print('mark: shuffle completed')

def vectorize_imgs(img_path_list):
    image_arr_list = []
    for img_path in img_path_list:
        
        if os.path.exists(BASE_PATH + img_path):
            img = Image.open(BASE_PATH + img_path)
           
            img_arr = np.asarray(img, dtype='float32')
            
        
            image_arr_list.append(img_arr)
        else:
            print(img_path)
    return image_arr_list


def get_batch_image_path(left_train, similar_train, start):
    end = (start + batch_size) % len(similar_train)
    if start < end:
        return left_train[start:end], similar_train[start:end], end
    # 当 start > end 时，从头返回
    return np.concatenate([left_train[start:], left_train[:end]]), \
           np.concatenate([similar_train[start:], similar_train[:end]]), \
           end


def get_batch_image_array(batch_left,  batch_similar):
    batch_similar = np.asarray(batch_similar)[:, np.newaxis]
    eye = np.eye(5005)              #生成对角矩阵，为后面生成onehot码做准备
    dev_onehot = []
    for arg in batch_similar:
        dev_onehot.append(eye[(arg)[0]-1])
        
    eye = []                #释放eye的空间
    dev_onehot = np.asarray(dev_onehot,dtype='float32')
    
    return np.asarray(vectorize_imgs(batch_left), dtype='float32') / 255., \
           dev_onehot


if __name__ == '__main__':
    a = []
    a.append('1.jpg')
    a.append('2.jpg')
    arr = vectorize_imgs(a)
    print(arr)