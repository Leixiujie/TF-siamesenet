import numpy as np
from PIL import Image
from config import FLAGS
import os
import gc

DEV_NUMBER = FLAGS.DEV_NUMBER
BASE_PATH = FLAGS.BASE_PATH
batch_size = FLAGS.batch_size

if not os.path.exists(FLAGS.negative_file):
        os.makedirs(FLAGS.negative_file)
        
negative_pairs_path_file = open(FLAGS.negative_file, 'r')
negative_pairs_path_lines = negative_pairs_path_file.readlines()
positive_pairs_path_file = open(FLAGS.positive_file, 'r')
positive_pairs_path_lines = positive_pairs_path_file.readlines()
print(len(negative_pairs_path_lines))
print(len(positive_pairs_path_lines))

print('mark: loaded positive_negative files')
left_image_path_list = []
right_image_path_list = []
similar_list = []

for line in negative_pairs_path_lines:
    left_right = line.strip().split(' ')
    left_image_path_list.append(left_right[0])
    right_image_path_list.append(left_right[1])
    similar_list.append(0)
gc.collect(negative_pairs_path_lines)


for line in positive_pairs_path_lines:
    left_right = line.strip().split(' ')
    left_image_path_list.append(left_right[0])
    right_image_path_list.append(left_right[1])
    similar_list.append(1)  
gc.collect(positive_pairs_path_lines)

print('mark: added positve and negative array ')
left_image_path_list = np.asarray(left_image_path_list)
right_image_path_list = np.asarray(right_image_path_list)
similar_list = np.asarray(similar_list)

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(similar_list)))
left_shuffled = left_image_path_list[shuffle_indices]
right_shuffled = right_image_path_list[shuffle_indices]
similar_shuffled = similar_list[shuffle_indices]

left_train = left_shuffled[:DEV_NUMBER]
left_dev = left_shuffled[DEV_NUMBER:]
right_train = right_shuffled[:DEV_NUMBER]
right_dev = right_shuffled[DEV_NUMBER:]
similar_train, similar_dev = similar_shuffled[:DEV_NUMBER], similar_shuffled[DEV_NUMBER:]
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


def get_batch_image_path(left_train, right_train, similar_train, start):
    end = (start + batch_size) % len(similar_train)
    if start < end:
        return left_train[start:end], right_train[start:end], similar_train[start:end], end
    # 当 start > end 时，从头返回
    return np.concatenate([left_train[start:], left_train[:end]]), \
           np.concatenate([right_train[start:], right_train[:end]]), \
           np.concatenate([similar_train[start:], similar_train[:end]]), \
           end


def get_batch_image_array(batch_left, batch_right, batch_similar):
    return np.asarray(vectorize_imgs(batch_left), dtype='float32') / 255., \
           np.asarray(vectorize_imgs(batch_right), dtype='float32') / 255., \
           np.asarray(batch_similar)[:, np.newaxis]


if __name__ == '__main__':
    a = []
    a.append('1.jpg')
    a.append('2.jpg')
    arr = vectorize_imgs(a)
    print(arr)