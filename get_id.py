import os
import numpy as np

base_path = './datas/distance_file/'
pair_paths_base = './datas/test_pair_lists/'


def find_biggest_arr(txt_name):
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

def get_idds(idd,pair_paths,txt_name):
    f = open(pair_paths,'r')
    pairs = f.readlines()
    txt_path = os.path.join(base_path,str(txt_name).strip())
    f = open(txt_path,'r')

    distance = f.readlines()
    for i in idd:
        print(pairs[i])
        print(distance[i])


if __name__ == '__main__':
    txt_name = '0aaa29830.txt'
    idd = find_biggest_arr(txt_name)
    pair_paths = os.path.join(pair_paths_base,txt_name)
    get_idds(idd,pair_paths,txt_name)
    print(idd)





