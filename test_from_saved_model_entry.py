from model import SIAMESE
import tensorflow as tf
from PIL import Image
import logging
import os
import numpy as np


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                        datefmt='%b %d %H:%M')
    
    with tf.name_scope("in"):
        left = tf.placeholder(tf.float32, [None, 72, 72, 3], name='left')
        right = tf.placeholder(tf.float32, [None, 72, 72, 3], name='right')
    with tf.name_scope("similarity"):
        label = tf.placeholder(tf.int32, [None, 1], name='label')  # 1 if same, 0 if different
        label = tf.to_float(label)
    
    left_output = SIAMESE().siamesenet(left, reuse=False)
    
    right_output = SIAMESE().siamesenet(right, reuse=True)
    
    model1, model2, distance, loss = SIAMESE().contrastive_loss(left_output, right_output, label)
    
    global_step = tf.Variable(0, trainable=False)
    
    
    
    train_step = tf.train.AdamOptimizer(0.0000001).minimize(loss, global_step=global_step) #小数点后7个0
    print("the model has been built")
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
        saver.restore(sess, './checkpoint/model_1000.ckpt')                  #此处输入要续接的模型
        
        
        base_path = './datas/'
        pairs_path = os.path.join(base_path,'test_pair_lists')
        for path,dirs,files in os.walk(pairs_path):
            break
        
        distance_file = os.path.join(base_path,'distance_file/')
        
        
        if not os.path.exists(distance_file):
            os.makedirs(distance_file)
            
        num_of_test_pic = 0
        time1 = 0
        for txt in files:
            num_of_test_pic += 1
            txt_path = os.path.join(pairs_path,str(txt).strip())
            comparison_file = open(txt_path, 'r')
            comparison_pairs_path_line = comparison_file.readlines()
            
            distance_txt_file = os.path.join(distance_file,str(txt).strip())
            f1 = open(distance_txt_file,'w')
            num_of_pics = 0
            output = ''
            left_pic_arrs = []
            right_pic_arrs = []
            for line in comparison_pairs_path_line:
                
                left_right = line.strip().split(' ')
                left_pic = Image.open(left_right[0])
                left_pic_arr = np.asarray(left_pic)/255.0
                left_pic_arr = left_pic_arr.reshape((72,72,3))
                
                left_pic_arrs.append(left_pic_arr)
                
                right_pic = Image.open(left_right[1])
                right_pic_arr = np.asarray(right_pic)/255.0
                right_pic_arr = right_pic_arr.reshape((72,72,3))
                right_pic_arrs.append(right_pic_arr)
                
                if num_of_pics % 100 ==0 or num_of_pics == 25360 and num_of_pics !=0:
                    
                    output_distance = sess.run([distance], feed_dict={left: left_pic_arrs, right: right_pic_arrs})
                    for some in output_distance[0]:
                        output = output + str(some[0]) +'\n'
                    left_pic_arrs = []
                    right_pic_arrs = []
                    
                num_of_pics += 1
                if(num_of_pics % 1000 ==0 or num_of_pics == 25360):
                    print('现在正在处理测试集的第'+str(num_of_test_pic)+'/7960张图片的第'+str(num_of_pics)+'/25361个对比组')
            print(len(output.strip().split('\n')))
            f1.write(output)
            f1.close()
        
if __name__ == '__main__' :
    main()
    