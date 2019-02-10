import tensorflow as tf

variables_dict = {
    "hidden_Weights_1": tf.Variable(tf.truncated_normal([144, 768], stddev=0.1), name="hidden_Weights_1"),
    "hidden_biases_1": tf.Variable(tf.constant(0.1, shape=[768]), name="hidden_biases_1"),
    "hidden_Weights_2": tf.Variable(tf.truncated_normal([768, 5005], stddev=0.1), name="hidden_Weights_2"),
    "hidden_biases_2": tf.Variable(tf.constant(0.1, shape=[5005]), name="hidden_biases_2")
}


class SIAMESE(object):
    def siamesenet(self, input, reuse=False):
        with tf.name_scope("model"):
            with tf.variable_scope("conv1") as scope:
                conv1 = tf.layers.conv2d(input, filters=64, kernel_size=[5, 5], strides=[1, 1],
                                         padding='SAME', activation=tf.nn.relu, reuse=reuse, name=scope.name)
                pool1 = tf.layers.max_pooling2d(conv1, pool_size=[3, 3], strides=[2, 2],
                                                padding='SAME', name='pool1')

            with tf.variable_scope("conv2") as scope:
                conv2 = tf.layers.conv2d(pool1, filters=128, kernel_size=[5, 5], strides=[1, 1],
                                         padding='SAME', activation=tf.nn.relu, reuse=reuse, name=scope.name)
                pool2 = tf.layers.max_pooling2d(conv2, pool_size=[3, 3], strides=[2, 2],
                                                padding='SAME', name='pool2')

            with tf.variable_scope("conv3") as scope:
                conv3 = tf.layers.conv2d(pool2, filters=256, kernel_size=[3, 3], strides=[1, 1],
                                         padding='SAME', activation=tf.nn.relu, reuse=reuse, name=scope.name)
                pool3 = tf.layers.max_pooling2d(conv3, pool_size=[3, 3], strides=[2, 2],
                                                padding='SAME', name='pool3')

            with tf.variable_scope("conv4") as scope:
                conv4 = tf.layers.conv2d(pool3, filters=512, kernel_size=[3, 3], strides=[1, 1],
                                         padding='SAME', activation=tf.nn.relu, reuse=reuse, name=scope.name)
                pool4 = tf.layers.max_pooling2d(conv4, pool_size=[3, 3], strides=[2, 2],
                                                padding='SAME', name='pool4')

            with tf.variable_scope("conv5") as scope:
                conv5 = tf.layers.conv2d(pool4, filters=16, kernel_size=[3, 3], strides=[1, 1],
                                         padding='SAME', activation=tf.nn.relu, reuse=reuse, name=scope.name)
                pool5 = tf.layers.max_pooling2d(conv5, pool_size=[3, 3], strides=[2, 2],
                                                padding='SAME', name='pool5')

            flattened = tf.contrib.layers.flatten(pool5)

            with tf.variable_scope("local1") as scope:
                output = tf.nn.leaky_relu(tf.matmul(flattened, variables_dict["hidden_Weights_1"]) +
                                    variables_dict["hidden_biases_1"],alpha = 0.2,name=scope.name)
            
                
        return output

    def contrastive_loss(self, output, y):
        with tf.name_scope("output"):
            y_ = tf.matmul(output, variables_dict["hidden_Weights_2"]) + variables_dict["hidden_biases_2"]

        with tf.name_scope("evalution"):
            correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
            evalution_step = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            
            
        # CalculateMean loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_)
            loss = tf.reduce_sum(losses)

        return  y_,evalution_step,loss
