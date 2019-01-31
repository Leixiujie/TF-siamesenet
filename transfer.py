import tensorflow as tf


ver = tf.train.import_meta_graph("./checkpoint/transfer/model_69000.ckpt.meta")
with tf.Graph().as_default() as g:
    saver = tf.train.import_meta_graph('./checkpoint/transfer/model_69000.ckpt.meta')
    x_left = g.get_tensor_by_name('in/left:0')
    x_right = g.get_tensor_by_name('in/right:0')
    y = g.get_tensor_by_name('similarity')
    weight_test = g.get_tensor_by_name('layer1/weights:0')
    layer2 = g.get_tensor_by_name('layer2/layer2:0')
    layer2 = tf.stop_gradient(layer2,name='stop_gradient')
