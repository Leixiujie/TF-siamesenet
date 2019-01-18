import tensorflow as tf
import model


def restore_model():
    with tf.Graph().as_default() as tg:
        left = tf.placeholder(tf.float32, [None, 72, 72, 3], name='left')
        right = tf.placeholder(tf.float32, [None, 72, 72, 3], name='right')
        distance = 