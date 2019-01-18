import normal               #把图片规则化的程序
import positive_negative    #正负样本对生成程序
import train                #训练程序
import time

#注意一定要是排好序并且去掉id和图片名称那一行的csv表格文件

csv_path = 'train.csv'
root = './train/'
transformed_images = './new_train/'

if __name__ =='__main__' :
    
    print('------------正负样本对生成------------')
    time.sleep(2)
    positive_negative.main()
    print('----------正负样本对生成完成----------')
    print(' ')
    
    print('------------图片标准化处理------------')  
    time.sleep(2)
    normal.all_pics_processing(root , transformed_images)
    print('----------图片标准化处理完成----------')
    
    print('---------------训练开始---------------')
    time.sleep(2)
    train.main()
    
    #若已经完成正负样本对和标准化处理可以删掉前两个，直接开始训练。