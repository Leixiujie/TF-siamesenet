import os
import csv
import xlrd

base_path = './datas/distance_file/'
pair_paths_base = './datas/test_pair_lists/'


def find_biggest_arr(files):
    idds = []
    tt = 0
    for txt_name in files:
        txt_path = os.path.join(base_path,str(txt_name).strip())
        f = open(txt_path,'r')
        distances = f.readlines()
    
        idd = [0,1,2,3,4]
        num = 5
        
        for distance in distances[5:]:
            minn,minn_index = get_minn(distances,idd)
            if(minn < eval(distance)):
                idd[minn_index] = num
            num += 1
        f.close()
        idds.append(idd)
        if (tt % 10 == 0):
            print('正在找第'+str(tt)+'/7960个图片的最大五个可能id编号')
        tt += 1
    print(len(idds))
    
    return idds

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

def get_idds(idds,fies):
    
    #打开编号对应id文件
    workbook = xlrd.open_workbook('train.csv')
    imginfo = workbook.sheet_by_index(0)
    ids = imginfo.col_values(1)
    #打开要写的csv文件
    f = open('example.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(('Image','Id'))
    tt = 0
    for txt_name in files:
        pair_paths = os.path.join(pair_paths_base,str(txt_name).strip())
    
    
    
        f = open(pair_paths,'r')
        pairs = f.readlines()
        f.close()
        txt_path = os.path.join(base_path,str(txt_name).strip())
        f = open(txt_path,'r')
        distance = f.readlines()
        f.close()
        
        idd = idds[tt]
        #排序 将他们的distance大小从小到大排出
        for i in range(5):
            for j in range(i+1,5):
                if(eval(distance[idd[i]]) < eval(distance[idd[j]])):
                    t = idd[i]
                    idd[i] = idd[j]
                    idd[j] = t
    
    
    
        #输出排序后的结果
        Id = ''
        for i in idd:
            Id =Id + str(ids[i]) + ' '
            Image = (pairs[i].strip().split(' ')[1]).split('/')[2]
    
        writer.writerow((Image,Id))
        tt += 1
        if (tt % 50 ==0):
            print('正在写第'+str(tt)+'/7960个图片的csv信息')
    
    
        
        
        

if __name__ == '__main__':
    for path,dirs,files in os.walk(base_path):
        break

    idds = find_biggest_arr(files)
    print(len(idds))
    get_idds(idds,files)






