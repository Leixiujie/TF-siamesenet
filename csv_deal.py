import xlrd
import os

def csv_deal(file_name):
    if not os.path.exists('./datas/'):
        os.makedirs('./datas/')
    
    workbook = xlrd.open_workbook(file_name)
    imginfo = workbook.sheet_by_index(0)
    
    path = imginfo.col_values(0)
    ids = imginfo.col_values(1)
    
    
    f1 = open("./datas/dataset_deal.txt",'w')
    f2 = open("./datas/dataset_id_num.txt",'w')  
    f3 = open("./datas/dataset_path.txt",'w')
    
    id_onehot = []
    iid = ids[0]
    id_onehot.append(iid)
    i = 0
    for _id in ids:
        if _id != iid:
            iid = _id
            id_onehot.append(iid)
        
        f1.write(str(path[i])+" "+str(len(id_onehot))+"\n")
        f2.write(str(len(id_onehot)) + " ")
        f3.write(str(path[i]) + " ")
        i = i + 1
        
    num_of_whale = len(id_onehot)
    print("数据中总共有"+str(num_of_whale)+"条鲸鱼(包括new_whale)")                        #输出总共id数（包括new_whale）
    
    
    f1.close()
    f2.close()
    f3.close()
    return path,ids,id_onehot

    
if __name__ == '__main__':
    path = 'train.csv'
    path,ids,id_onehot = csv_deal(path)
    