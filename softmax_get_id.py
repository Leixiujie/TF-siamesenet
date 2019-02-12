import xlrd
import csv
import os

def get_sort(distance,idd):
    for i in range(5):
        for j in range(i+1,5):
            if(eval(distance[idd[i]]) < eval(distance[idd[j]])):
                t = idd[i]
                idd[i] = idd[j]
                idd[j] = t
    return idd

def get_minn(distances,idd):
    minn = eval(distances[idd[0]])
    idd_index = 0
    t = 1
    for i in idd[1:5]:
        if minn > eval(distances[i]):
            minn = eval(distances[i])
            idd_index = t
        t += 1
    return minn,idd_index

def find_biggest_arr(distances):
    idd = [0,1,2,3,4]
    num = 5
        
    for distance in distances[5:]:
        minn,minn_index = get_minn(distances,idd)
        if(minn < eval(distance)):
            idd[minn_index] = num
        num += 1
    idd = get_sort(distances,idd)                #返回排过序的idd
    return idd

def get_max(file_name):
    f = open(file_name)
    distances = f.readlines()
    print(len(distances))
    text_row = 0
    idd = []
    idds = []
    while (text_row < int(len(distances))):          #len(distances)
        row = distances[text_row]
        row = row.strip().split(' ')
        idd = find_biggest_arr(row)
        idds.append(idd)
        text_row += 1
        if text_row % 10 == 0:
            print('当前正在处理第'+str(text_row)+'个图片的ID')
    return idds


    
    
    
def get_idds(idds):
    f = open('example.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(('Image','Id'))
    for path,dirs,files in os.walk('./test/'):
        break
    for path,dirs,train_files in os.walk('./new_train/'):
        break
    f2 = open('./datas/dataset_path.txt')
    paths = f2.readline()
    paths = paths.strip().split(' ')
    f2.close()
    f2 = open('./datas/softmax_name_list.txt')
    file_train = f2.readlines()
    '''
    workbook = xlrd.open_workbook('train.csv')     #从按图片排序的csv中读取
    imginfo = workbook.sheet_by_index(0)
    file_train = imginfo.col_values(1)               #此处file_train 应该为id集合
    '''
    file_num = 0
    for file in files[:int(len(idds))]:  
        what_to_write = ''
        idd = idds[file_num]

        for Id in idd:
            what_to_write = what_to_write + str(file_train[Id-1].strip()) +' '
        writer.writerow((file.strip(),what_to_write.strip()))
        file_num += 1
            

if __name__ == '__main__':
    txt_name = 'distance_all.txt'
    txt_name = os.path.join('./datas/distance_file/',txt_name)
    idds = get_max(txt_name)
    get_idds(idds)
