import normal
import os
import tensorflow as tf
from config import FLAGS
from dataset import *

root = './test/'
transformed_images = './new_test/'
transformed_train_images = './new_train/'

def test_pic_processing():
    normal.all_pics_processing(root , transformed_images)
    print('--------图片处理完成-------------')

def test_pairs_generation(test_images_path,train_images_path):
    lists_path = './datas/test_pair_lists/'
    if not os.path.exists(lists_path):
        os.makedirs(lists_path)
    
    for path,dirs,files in os.walk(test_images_path):
        break
    
    f = open('./datas/dataset_path.txt','r')
    paths = f.readline()
    paths = paths.strip().split(' ')
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
        for path in paths:
            f.write('./new_train/' + str(path) + ' ' + './new_test/' + str(file) + '\n')
        f.close()
    

def calculate_id(pairs_paths):
    #再次载入样本对 
    negative_pairs_path_file = open(FLAGS.negative_file, 'r')
    negative_pairs_path_lines = negative_pairs_path_file.readlines()
    positive_pairs_path_file = open(FLAGS.positive_file, 'r')
    positive_pairs_path_lines = positive_pairs_path_file.readlines()
    
    print('mark: loaded positive_negative files')
    left_image_path_list = []
    right_image_path_list = []
    similar_list = []
    
    for line in negative_pairs_path_lines:
        left_right = line.strip().split(' ')
        left_image_path_list.append(left_right[0])
        right_image_path_list.append(left_right[1])
        similar_list.append(0)
    
    for line in positive_pairs_path_lines:
        left_right = line.strip().split(' ')
        left_image_path_list.append(left_right[0])
        right_image_path_list.append(left_right[1])
        similar_list.append(1)
    print('mark: added positve and negative array ')
    left_image_path_list = np.asarray(left_image_path_list)
    right_image_path_list = np.asarray(right_image_path_list)
    similar_list = np.asarray(similar_list)
    
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(similar_list)))
    left_shuffled = left_image_path_list[shuffle_indices]
    right_shuffled = right_image_path_list[shuffle_indices]
    similar_shuffled = similar_list[shuffle_indices]
    
    left_dev = left_shuffled[0:30000]
    right_dev = right_shuffled[0:30000]
    similar_dev = similar_shuffled[0:30000]
    print('mark: shuffle completed')
    
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement = True,log_device_placement = False)
        sess = tf.Session(config = session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph('checkpointt/model_133000.ckpt.meta')
            saver.restore(sess, 'checkpointt/model_133000.ckpt')
            
            left = graph.get_operation_by_name("in/left").outputs[0]
            right = graph.get_operation_by_name("in/right").outputs[0]
            
            distance = graph.get_operation_by_name("output/distance").outputs[0]
            sigmoid_distance = tf.nn.sigmoid(distance)
            
            image_test = []
            label_test = []
            index = 1
            
            for i in range(30000/50):
                print('------validation here------')
                num_of_true = 0
                for t in range(0,abs(int(DEV_NUMBER / 50))):
                    val_distance_district = sess.run([distance,sigmoid_distance],
                                            feed_dict={left: left_dev_arr[50*t:50*(t+1),:,:,:], right: right_dev_arr[50*t:50*(t+1),:,:,:], label: similar_dev_arr[50*t:50*(t+1)]})
                    
                    index = 0
                    
                    for j in val_distance_district[0]:
                        if j >= 0:
                            j = 1
                        else:
                            j = 0
                        
                        if j ==similar_dev_arr[50*t:50*(t+1)][index] :
                            num_of_true += 1
                        index += 1
                
                print('测试集的正确率为：' + str(1.0*num_of_true/2000))

    
                    image_test = []
                    label_test = []

            
            
    
        
    

if __name__ == '__main__' :
    pairs_dir = './datas/test_pair_lists/'
    
    
    process = 3
    while(process != 0 and process != 1):
        process = eval(input('请输入是否初始化图片和产生对比对\n输入0表示不处理\n输入1表示处理\n请输入：'))
        if(process != 0 and process != 1):
            print('\n输入错误，请重新输入\n')
    
    if( process == 1):
        test_pic_processing()
        test_pairs_generation(transformed_images,transformed_train_images)
        
    for path,dirs,files in os.walk(pairs_dir):
        break
        
    