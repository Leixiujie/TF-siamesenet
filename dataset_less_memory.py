<<<<<<< HEAD
import numpy as np
from PIL import Image
from config import FLAGS
import os

DEV_NUMBER = FLAGS.DEV_NUMBER
BASE_PATH = FLAGS.BASE_PATH
batch_size = FLAGS.batch_size

        


print('mark: dividing the negative dataset')

def dataset_divide():
    negative_pairs_path_file = open(FLAGS.negative_file, 'r')
    negative_pairs_path_lines = negative_pairs_path_file.readlines()
    negative_pairs_path_file.close()
    positive_pairs_path_file = open(FLAGS.positive_file, 'r')
    positive_pairs_path_lines = positive_pairs_path_file.readlines()
    positive_pairs_path_file.close()
    print(len(negative_pairs_path_lines))
    print(len(positive_pairs_path_lines))
    
    left_image_path_list = []
    right_image_path_list = []
    similar_list = []
    for line in negative_pairs_path_lines:
        left_right = line.strip().split(' ')
        left_image_path_list.append(left_right[0])
        right_image_path_list.append(left_right[1])
        similar_list.append(0)
    negative_pairs_path_lines = []
    
    
    for line in positive_pairs_path_lines:
        left_right = line.strip().split(' ')
        left_image_path_list.append(left_right[0])
        right_image_path_list.append(left_right[1])
        similar_list.append(1)  
    positive_pairs_path_lines = []
    
    print('mark: added positve and negative array ')
    left_image_path_list = np.asarray(left_image_path_list)
    right_image_path_list = np.asarray(right_image_path_list)
    similar_list = np.asarray(similar_list)
    
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(similar_list)))
    left_shuffled = left_image_path_list[shuffle_indices]
    left_image_path_list = []
    right_shuffled = right_image_path_list[shuffle_indices]
    right_image_path_list = []
    similar_shuffled = similar_list[shuffle_indices]
    similar_list = []
    print('mark: shuffle completed')
    
    file_num = 0
    file_name = 'pairs_'+str(file_num).strip() +'.txt'
    pairs_shuffled_path = './datas/pairs/'
    if not os.path.exists(pairs_shuffled_path):
        os.makedirs(pairs_shuffled_path)
    file_path = os.path.join(pairs_shuffled_path,file_name)
    file_num += 1
    f = open(file_path,'w')
    for i in range(len(similar_shuffled)):
        if ((i % 2000000 == 0 ) and i != 0):
            f.close()
            file_name = 'pairs_'+str(file_num).strip()+'.txt'
            file_path = os.path.join(pairs_shuffled_path,file_name)
            f = open(file_path,'w')
            file_num += 1
        f.write(str(left_shuffled[i]).strip() + ' ' + str(right_shuffled[i]).strip() + ' ' + str(similar_shuffled[i]).strip() + '\n')
        
        if(i % 1000000 == 0):
            print(i)
            
        if(i == (len(similar_shuffled)-1)):
            f.close()

        
    

def get_dataset(pairs_file,devset_need_or_not = False):
    f = open(pairs_file,'r')
    pairs_path_lines = f.readlines()
    print('txt 长度是 '+str(len(pairs_path_lines)))
    f.close()
    
    
    left_image_path_list = []
    right_image_path_list = []
    similar_list = []
    for line in pairs_path_lines:
        left_right_similar = line.strip().split(' ')
        left_image_path_list.append(left_right_similar[0])
        right_image_path_list.append(left_right_similar[1])
        similar_list.append(eval(left_right_similar[2]))
    pairs_path_lines = []
    
    print('mark: added positve and negative array ')
    left_image_path_list = np.asarray(left_image_path_list)
    right_image_path_list = np.asarray(right_image_path_list)
    similar_list = np.asarray(similar_list)
    
    if devset_need_or_not :
        left_train = left_image_path_list[:DEV_NUMBER]
        left_dev = left_image_path_list[DEV_NUMBER:]
        left_image_path_list = []
        right_train = right_image_path_list[:DEV_NUMBER]
        right_dev = right_image_path_list[DEV_NUMBER:]
        right_image_path_list = []
        similar_train, similar_dev = similar_list[:DEV_NUMBER], similar_list[DEV_NUMBER:]
        similar_list = []
        print('mark: shuffle completed')
        print(len(left_train))
        print(len(right_train))
        print(len(left_dev))
        print(len(right_dev))
        print(len(similar_train))
        return left_train,left_dev,right_train,right_dev,similar_train,similar_dev
    else:
        return left_image_path_list,right_image_path_list,similar_list
    
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


def get_batch_image_path(left_train, right_train, similar_train, start):
    end = (start + batch_size) % len(similar_train)
    if start < end:
        return left_train[start:end], right_train[start:end], similar_train[start:end], end,0
    # 当 start > end 时，从头返回
    return np.concatenate([left_train[start:], left_train[:end]]), \
           np.concatenate([right_train[start:], right_train[:end]]), \
           np.concatenate([similar_train[start:], similar_train[:end]]), \
           end,1


def get_batch_image_array(batch_left, batch_right, batch_similar):
    return np.asarray(vectorize_imgs(batch_left), dtype='float32') / 255., \
           np.asarray(vectorize_imgs(batch_right), dtype='float32') / 255., \
           np.asarray(batch_similar)[:, np.newaxis]


if __name__ == '__main__':
=======
import numpy as np
from PIL import Image
from config import FLAGS
import os

DEV_NUMBER = FLAGS.DEV_NUMBER
BASE_PATH = FLAGS.BASE_PATH
batch_size = FLAGS.batch_size

        


print('mark: dividing the negative dataset')

def dataset_divide():
    negative_pairs_path_file = open(FLAGS.negative_file, 'r')
    negative_pairs_path_lines = negative_pairs_path_file.readlines()
    negative_pairs_path_file.close()
    positive_pairs_path_file = open(FLAGS.positive_file, 'r')
    positive_pairs_path_lines = positive_pairs_path_file.readlines()
    positive_pairs_path_file.close()
    print(len(negative_pairs_path_lines))
    print(len(positive_pairs_path_lines))
    
    left_image_path_list = []
    right_image_path_list = []
    similar_list = []
    for line in negative_pairs_path_lines:
        left_right = line.strip().split(' ')
        left_image_path_list.append(left_right[0])
        right_image_path_list.append(left_right[1])
        similar_list.append(0)
    negative_pairs_path_lines = []
    
    
    for line in positive_pairs_path_lines:
        left_right = line.strip().split(' ')
        left_image_path_list.append(left_right[0])
        right_image_path_list.append(left_right[1])
        similar_list.append(1)  
    positive_pairs_path_lines = []
    
    print('mark: added positve and negative array ')
    left_image_path_list = np.asarray(left_image_path_list)
    right_image_path_list = np.asarray(right_image_path_list)
    similar_list = np.asarray(similar_list)
    
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(similar_list)))
    left_shuffled = left_image_path_list[shuffle_indices]
    left_image_path_list = []
    right_shuffled = right_image_path_list[shuffle_indices]
    right_image_path_list = []
    similar_shuffled = similar_list[shuffle_indices]
    similar_list = []
    print('mark: shuffle completed')
    
    file_num = 0
    file_name = 'pairs_'+str(file_num).strip() +'.txt'
    pairs_shuffled_path = './datas/pairs/'
    if not os.path.exists(pairs_shuffled_path):
        os.makedirs(pairs_shuffled_path)
    file_path = os.path.join(pairs_shuffled_path,file_name)
    file_num += 1
    f = open(file_path,'w')
    for i in range(len(similar_shuffled)):
        if ((i % 2000000 == 0 ) and i != 0):
            f.close()
            file_name = 'pairs_'+str(file_num).strip()+'.txt'
            file_path = os.path.join(pairs_shuffled_path,file_name)
            f = open(file_path,'w')
            file_num += 1
        f.write(str(left_shuffled[i]).strip() + ' ' + str(right_shuffled[i]).strip() + ' ' + str(similar_shuffled[i]).strip() + '\n')
        
        if(i % 1000000 == 0):
            print(i)
            
        if(i == (len(similar_shuffled)-1)):
            f.close()

        
    

def get_dataset(pairs_file,devset_need_or_not = False):
    f = open(pairs_file,'r')
    pairs_path_lines = f.readlines()
    print('txt 长度是 '+str(len(pairs_path_lines)))
    f.close()
    
    
    left_image_path_list = []
    right_image_path_list = []
    similar_list = []
    for line in pairs_path_lines:
        left_right_similar = line.strip().split(' ')
        left_image_path_list.append(left_right_similar[0])
        right_image_path_list.append(left_right_similar[1])
        similar_list.append(eval(left_right_similar[2]))
    pairs_path_lines = []
    
    print('mark: added positve and negative array ')
    left_image_path_list = np.asarray(left_image_path_list)
    right_image_path_list = np.asarray(right_image_path_list)
    similar_list = np.asarray(similar_list)
    
    if devset_need_or_not :
        left_train = left_image_path_list[:DEV_NUMBER]
        left_dev = left_image_path_list[DEV_NUMBER:]
        left_image_path_list = []
        right_train = right_image_path_list[:DEV_NUMBER]
        right_dev = right_image_path_list[DEV_NUMBER:]
        right_image_path_list = []
        similar_train, similar_dev = similar_list[:DEV_NUMBER], similar_list[DEV_NUMBER:]
        similar_list = []
        print('mark: shuffle completed')
        print(len(left_train))
        print(len(right_train))
        print(len(left_dev))
        print(len(right_dev))
        print(len(similar_train))
        return left_train,left_dev,right_train,right_dev,similar_train,similar_dev
    else:
        return left_image_path_list,right_image_path_list,similar_list
    
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


def get_batch_image_path(left_train, right_train, similar_train, start):
    end = (start + batch_size) % len(similar_train)
    if start < end:
        return left_train[start:end], right_train[start:end], similar_train[start:end], end,0
    # 当 start > end 时，从头返回
    return np.concatenate([left_train[start:], left_train[:end]]), \
           np.concatenate([right_train[start:], right_train[:end]]), \
           np.concatenate([similar_train[start:], similar_train[:end]]), \
           end,1


def get_batch_image_array(batch_left, batch_right, batch_similar):
    return np.asarray(vectorize_imgs(batch_left), dtype='float32') / 255., \
           np.asarray(vectorize_imgs(batch_right), dtype='float32') / 255., \
           np.asarray(batch_similar)[:, np.newaxis]


if __name__ == '__main__':
>>>>>>> b85eb4c5a75f8b1ede29249e0b75d2afbe729869
    dataset_divide()