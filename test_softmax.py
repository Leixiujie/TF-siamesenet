import tensorflow as tf
import numpy as np
import xlrd
from softmax_model import SIAMESE
from dataset_softmax import get_batch_image_path,get_batch_image_array
import logging
from PIL import Image
import time
import os

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                        datefmt='%b %d %H:%M')
    
    with tf.name_scope("in"):
        x = tf.placeholder(tf.float32, [None, 72, 72, 3], name='left')
    with tf.name_scope("similarity"):
        label = tf.placeholder(tf.float32, [None, 5005], name='label')  
    
    output = SIAMESE().siamesenet(x, reuse=False)
    output_max = tf.argmax(output,1)


    output,evalution_step,loss = SIAMESE().contrastive_loss(output, label)
    
    global_step = tf.Variable(0, trainable=False)
    
    #定义需要训练的层
    var = tf.global_variables()
    train_step = tf.train.AdamOptimizer(0.00001).minimize(loss, global_step=global_step)
    print("the model has been built")
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        var = tf.global_variables()
        
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var, max_to_keep=20)
        restore_file_name_path = './softmax_checkpoint/model_404000.ckpt'                #此处输入要续接的模型
        
        saver.restore(sess, restore_file_name_path)                 
    
        
        # train iter
        idx = 0
        i = 0
        jump_out_or_not = 1                     #此处为0跳出循环
        #while (jump_out_or_not == 1):
        for i in range(1):
            distance_file = './datas/distance_file/'
            if not os.path.exists(distance_file):
                os.makedirs(distance_file)
            
            distance_txt_file = os.path.join(distance_file,'distance_all.txt')
            f1 = open(distance_txt_file,'w')
            img_paths = []
            
            for left_path,left_dirs,left_files in os.walk('./test_test/'):
                break
            
            
            '''
            batch_paths, batch_id_num, idx, jump_out_or_not = get_batch_image_path(left_files,start = idx,train_or_test = False)
            batch_img_arr, batch_id_num_arr = get_batch_image_array(batch_paths, batch_id_num)
            '''
            path = './new_train/0_0000e88ab.jpg'
            img = Image.open(path)
            img_arr = np.asarray(img,dtype='float32')/255.0
            img_arr = img_arr.reshape((1,72,72,3))
            output,output_max = sess.run([output,output_max],feed_dict={x:img_arr})
            print(output_max)
            #output = sess.run(sess.run([output],feed_dict={x:batch_img_arr}))

                  
    
            if (i % 100 == 0): 
                print('\n正在处理'+str(i)+'张图片\n')
            i += 1

if __name__ == '__main__' :
    main()