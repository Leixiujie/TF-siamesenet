import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('train_iter', 1000000, 'Total training iter')
flags.DEFINE_integer('validation_step', 50, 'Total training iter')
flags.DEFINE_integer('step', 1000, 'Save after ... iteration')
flags.DEFINE_integer('DEV_NUMBER', -2000, '验证集数量')
flags.DEFINE_integer('batch_size', 128, '批大小')
flags.DEFINE_string('BASE_PATH', './new_train/', '图片位置')
flags.DEFINE_string('negative_file', './datas/negative_pairs_path.txt', '不同人的文件')
flags.DEFINE_string('positive_file', './datas/positive_pairs_path.txt', '相同人的文件')
