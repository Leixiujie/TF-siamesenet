from model import SIAMESE
import tensorflow as tf
from PIL import Image
import logging
import os
import numpy as np
import time

compare_batch = 50

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
        
        distance_file = os.path.join(base_path,'distance_file/')
        
        
        if not os.path.exists(distance_file):
            os.makedirs(distance_file)
            
        for txt in range(1):
            distance_txt_file = os.path.join(distance_file,'distance_all.txt')
            f1 = open(distance_txt_file,'w')
            left_pic_arrs = []
            right_pic_arrs = []
            left_path = './new_test/'
            right_path = './new_train/'
            
            for left_path,left_dirs,left_files in os.walk(left_path):
                break
            
            right_filess = open('./datas/dataset_path.txt','r')
            right_files = right_filess.readline()
            right_filess.close()
            right_files = right_files.strip().split(' ')
            
            
            for img in left_files:
                 img_path = os.path.join(left_path,str(img).strip())
                 image = Image.open(img_path)
                 image_arr = np.asarray(image)/255.0
                 image_arr = image_arr.reshape((72,72,3))
                 left_pic_arrs.append(image_arr)
            
            
            for img in right_files:
                img_path = os.path.join(right_path,str(img).strip())
                image = Image.open(img_path)
                image_arr = np.asarray(image)/255.0
                image_arr = image_arr.reshape((72,72,3))
                right_pic_arrs.append(image_arr)
                
            num_of_test_now = 0
            for left_img_arr in left_pic_arrs:
                left_arrs_0 = []                       #用于前面的能凑够compare_batch的分组
                left_arrs_1 = []                       #用于尾部那无法凑够compare_batch的配对
                for i in range(compare_batch):
                    left_arrs_0.append(left_img_arr)
                for i in range(25361 % compare_batch):
                    left_arrs_1.append(left_img_arr)
                
                print(len(left_arrs_0))
                print(len(left_arrs_1))
                
                
                output = ''
                validation_iteration = 0
                time0 = time.time()
                while(validation_iteration < (int(25361/compare_batch)+1)):          #把每一个test图片和train的图片跑一遍
                    iii = 0
                    if 25361 - validation_iteration*compare_batch >= compare_batch:
                                                
                        right_arrs_0 = right_pic_arrs[validation_iteration*compare_batch:(validation_iteration+1)*compare_batch]
                        output_distance = sess.run([distance], feed_dict={left:left_arrs_0 ,
                                                       right: right_arrs_0})
                    else:
                        right_arr_1 = right_pic_arrs[validation_iteration*compare_batch:]
                        output_distance = sess.run([distance], feed_dict={
                                left: left_arrs_1, 
                                right: right_arr_1})
                    
                    for some in output_distance[0]:
                        output = output + str(some[0]) +' '
                    validation_iteration += 1
                    if validation_iteration % 5 == 0 :
                        print('当前处于第'+str(num_of_test_now)+'/7960测试张图片的'+str(validation_iteration*compare_batch)+'/25361张')
                    
                output = output +'\n\n\n'
                f1.write(output)
                num_of_test_now += 1
                
        f1.close()
        
if __name__ == '__main__' :
    main()
    