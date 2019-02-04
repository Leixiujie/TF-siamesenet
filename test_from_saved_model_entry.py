from model import SIAMESE
import tensorflow as tf
from PIL import Image
import xlrd
import logging
import os
import numpy as np
import time

compare_batch = 320                     #compare batch尽量取10的倍数，不要让25361整除

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
        saver.restore(sess, './checkpoint/model_151000.ckpt')                  #此处输入要续接的模型
        
        
        base_path = './datas/'
        
        distance_file = os.path.join(base_path,'distance_file/')
        
        
        if not os.path.exists(distance_file):
            os.makedirs(distance_file)
        '''          
        此段是单独测试某两张图片是否是同一条鲸鱼的程序  
        while(True):
            left_pic = (input("请输入左边样本（单个）")).strip()
            img_path = os.path.join('./new_train/',left_pic)
            left_img = Image.open(img_path)
            image_arr = np.asarray(left_img,dtype='float32')/255.
            left_image_arr = image_arr.reshape((1,72,72,3))
            
            right_pic = (input("请输入右边样本（单个）")).strip()
            img_path = os.path.join('./new_train/',right_pic)
            right_img = Image.open(img_path)
            image_arr = np.asarray(right_img,dtype='float32')/255.
            right_image_arr = image_arr.reshape((1,72,72,3))
            
            output_distance = sess.run([distance], feed_dict={left:left_image_arr ,right: right_image_arr})
            print(output_distance)
            print(output_distance[0][0])
        '''            
            
            
        
        for txt in range(1):
            distance_txt_file = os.path.join(distance_file,'distance_all.txt')
            f1 = open(distance_txt_file,'w')
            left_pic_arrs = []
            right_pic_arrs = []
            left_path = './test_test/'
            right_path = './test_train/'
            
            for left_path,left_dirs,left_files in os.walk(left_path):
                break
            
            workbook = xlrd.open_workbook('train.csv')     #从按图片排序的csv中读取
            imginfo = workbook.sheet_by_index(0)
            right_files = imginfo.col_values(0)
            
            
            for img in left_files:
                 img_path = os.path.join(left_path,str(img).strip())
                 image = Image.open(img_path)
                 image_arr = np.asarray(image,dtype='float32')/255.
                 image_arr = image_arr.reshape((72,72,3))
                 left_pic_arrs.append(image_arr)
            
            
            for img in right_files:
                img_path = os.path.join(right_path,str(img).strip())
                image = Image.open(img_path)
                image_arr = np.asarray(image,dtype='float32')/255.
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
                    if (25361 - validation_iteration*compare_batch) >= compare_batch:
                                                
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
                    if validation_iteration % 10 == 0 :
                        print('当前处于第'+str(num_of_test_now)+'/7960测试张图片的'+str(validation_iteration*compare_batch)+'/25361张')
                    
                output = output +'\n\n\n'
                f1.write(output)
                num_of_test_now += 1
                if (num_of_test_now % 10 ==0):                        #每10组写进txt一次
                    f1.close()
                    f1 = open(distance_txt_file,'a+')
        f1.close()
       
if __name__ == '__main__' :
    main()
    