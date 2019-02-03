import normal
import positive_genative
test_root = './test/'
test_transformed_images = './test_test/'

train_root = './train'
train_transformed_images = './test_train'

def test_pic_processing():
    normal.all_pics_processing(test_root , test_transformed_images,enhancement = False)
    print('\n\n--------测试图片处理完成-------------\n\n')
    print('\n\n--------正在处理训练集图片（无数据加强）-------------\n\n')
    normal.all_pics_processing(train_root , train_transformed_images,enhancement = False)
    print('\n\n--------训练集图片处理完成-------------\n\n')


if __name__ == '__main__':
    test_pic_processing()