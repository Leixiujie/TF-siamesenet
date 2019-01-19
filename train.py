import tensorflow as tf
from model import SIAMESE
from dataset import *
import logging

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
        saver.restore(sess, './checkpoint/model_49000.ckpt')                  #此处输入要续接的模型
    
        # setup tensorboard
        tf.summary.scalar('step', global_step)
        tf.summary.scalar('loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('train.log', sess.graph)
    
        left_dev_arr, right_dev_arr, similar_dev_arr = get_batch_image_array(left_dev, right_dev, similar_dev)
        # train iter
        idx = 0
        
        for i in range(FLAGS.train_iter):
            
            batch_left, batch_right, batch_similar, idx = get_batch_image_path(left_train, right_train, similar_train, idx)
            batch_left_arr, batch_right_arr, batch_similar_arr = \
                get_batch_image_array(batch_left, batch_right, batch_similar)
    
            steps, l, summary_str = sess.run([train_step, loss, merged],
                                         feed_dict={left: batch_left_arr, right: batch_right_arr, label: batch_similar_arr})
            print('after '+str(i) + ' times training,loss is ' + str(l))
            writer.add_summary(summary_str, i)
    
            #测试正确率
            if (i + 1) % FLAGS.validation_step == 0:
                print('------validation here------')
                num_of_true = 0
                for t in range(0,abs(int(DEV_NUMBER / 50))):
                    val_distance_district = sess.run([distance],
                                            feed_dict={left: left_dev_arr[50*t:50*(t+1),:,:,:], right: right_dev_arr[50*t:50*(t+1),:,:,:]})
                    index = 0
                    
                    for j in val_distance_district[0]:
                        if j >= 0.5:
                            j = 1
                        else:
                            j = 0
                        
                        if j ==similar_dev_arr[50*t:50*(t+1)][index] :
                            num_of_true += 1
                        index += 1
                
                print('测试集的正确率为：' + str(1.0*num_of_true/2000))
                        
                
    
            if i % FLAGS.step == 0 and i != 0:
                saver.save(sess, "checkpoint/model_%d.ckpt" % i)

if __name__ == '__main__' :
    main()