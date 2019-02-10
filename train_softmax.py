import tensorflow as tf
import numpy as np
from softmax_model import SIAMESE
from dataset_softmax import *
import logging
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

    
    output,evalution_step,loss = SIAMESE().contrastive_loss(output, label)
    
    global_step = tf.Variable(0, trainable=False)
    
    #定义需要训练的层
    var = tf.global_variables()
    #var_to_train = [val  for val in var if not 'conv1' in val.name or not 'conv2' in val.name or not 'conv3' in val.name or not 'conv4' in val.name or not 'conv5' in val.name]
    
    #train_step = tf.train.AdamOptimizer(0.00001).minimize(loss, global_step=global_step, var_list = var_to_train) #小数点后7个0
    train_step = tf.train.AdamOptimizer(0.00001).minimize(loss, global_step=global_step)
    print("the model has been built")
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        var = tf.global_variables()
        
        '''
        此处用作提取之前训练好的提取特征的卷积网络
        第一次读入之前训练好的参数时使用
        var_to_restore = [val  for val in var \
                          if 'conv1' in val.name \
                          or 'conv2' in val.name \
                          or 'conv3' in val.name \
                          or 'conv4' in val.name \
                          or 'conv5' in val.name]
        '''
        
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var, max_to_keep=20)
        restore_file_name_path = './softmax_checkpoint/model_312400.ckpt'
        
        saver.restore(sess, restore_file_name_path)                  #此处输入要续接的模型
        start = int(eval((((restore_file_name_path.strip().split('_'))[2]).strip().split('.'))[0]))
        saver.restore(sess, restore_file_name_path)                  #此处输入要续接的模型
    
        # setup tensorboard
        tf.summary.scalar('step', global_step)
        tf.summary.scalar('loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('train.log', sess.graph)
        
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
    
        dev_arrs,  dev_id_nums = get_batch_image_array(dev_paths, dev_id_num)
        '''
        此段为临时测试，看看下摆是否对齐
        for i in range(3):
            place = np.random.randint(0,2000)
            print(dev_id_nums[place])
            print(dev_id_num[place])
            maxx = dev_id_nums[place][0]
            index = 0
            where = 0
            for t in dev_id_nums[place]:
                index += 1
                if maxx<t:
                    maxx = t
                    where = index
            print(where)
        
        time.sleep(20)
        '''
        
        # train iter
        idx = 0
        acc_before = 0
        acc = 0
        for i in range(start,FLAGS.train_iter):
            
            batch_paths, batch_id_num, idx = get_batch_image_path(train_paths, train_id_num, idx)
            batch_img_arr, batch_id_num_arr = \
                get_batch_image_array(batch_paths, batch_id_num)
    
            steps, l,acce, summary_str = sess.run([train_step, loss,evalution_step, merged],
                                         feed_dict={x:batch_img_arr, label: batch_id_num_arr})
            print('after '+str(i) + ' times training,loss is ' + str(l))
            print('当前喂入batch正确为'+str(acce))
            writer.add_summary(summary_str, i)
            
            #测试正确率
            if (i + 1) % FLAGS.validation_step == 0:
                print('\n------validation here------')
                num_of_true = 0
                all_precision = 0
                for t in range(0,abs(int(DEV_NUMBER / 50))):
                    precision_district = sess.run([evalution_step],
                                            feed_dict={x: dev_arrs[50*t:50*(t+1),:,:,:], label: dev_id_nums[50*t:50*(t+1),:]})
                    all_precision += precision_district[0]
                
                acc = all_precision/40.0
                print('测试集的正确率为：' + str(acc)+' acc_before 是 '+ str(acc_before))
                print('------validation over------\n')
                        
                
    
            if (i % 100 == 0) and (i != start) and (acc >= acc_before):
                if not os.path.exists('./softmax_checkpoint/'):
                    os.makedirs('./softmax_checkpoint/')
                saver.save(sess, "./softmax_checkpoint/model_%d.ckpt" % i)
                acc_before = acc
                print('\nnew  model saved\n')

if __name__ == '__main__' :
    main()