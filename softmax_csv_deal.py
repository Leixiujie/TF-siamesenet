import csv
import os

def csv_deal(file_name):
    if not os.path.exists('./datas/'):
        os.makedirs('./datas/')
    
    f1 = open("./datas/softmax_path.txt",'w')
    
    workbook = csv.reader(open(file_name,'r'))
    ids = []
    path = []
    for info in workbook:
        path.append(info[0])
        ids.append(info[1])
    
    
    id_onehot = []
    iid = ids[0]
    id_onehot.append(iid)
    i = 1
    
    for _id in ids[1:]:
        if _id != iid:
            iid = _id
            id_onehot.append(iid)
        
        f1.write(str(path[i])+" "+str(len(id_onehot))+"\n")

        i = i + 1
        
    num_of_whale = len(id_onehot)
    print("数据中总共有"+str(num_of_whale)+"条鲸鱼(包括new_whale)")                        #输出总共id数（包括new_whale）
    
    
    f1.close()

    return path,ids,id_onehot
    
    
if __name__ == '__main__':
    path = 'softmax_dataset.csv'
    csv_deal(path)
    