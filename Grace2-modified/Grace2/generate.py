import math
import shutil
import os
import numpy as np
def getrandom(root_path,folderpath):
    '''
    :param root_path:folder_path的上级目录，方便存放分好的数据集，
           folder_path：所有的数据集存放的目录
    :return:
    '''
    project_names = []
    for project_name in os.listdir(folderpath):
        project_names.append(project_name)
    datalen= len(project_names)
    validlen=math.floor(datalen/10)*10
    if validlen!=datalen:
        invalid_project_index = np.random.choice(datalen, datalen - validlen, replace=False)
        for invalid_index in invalid_project_index:
            del project_names[invalid_index]
        # 随机选取项目删除，使最后能分的数据集长度为10的倍数
        datalen = len(project_names)  # 更新数据集长度
    valid_project_index=np.random.choice(datalen,datalen,replace=False)
    project_names_sliced=[]
    tem_project_names = []
    for i in range(datalen):
        tem_project_names.append(project_names[valid_project_index[i]])
        if i%(datalen/10)==0 and i>0:
            project_names_sliced.append(tem_project_names)
            tem_project_names=[]
    project_names_sliced.append(tem_project_names)
    os.mkdir('./datasliced')
    for i in range(10):
        group_path=root_path+'/datasliced'+'/{}'.format(i)
        os.mkdir(group_path)
        for j in project_names_sliced[i]:
            tem_path=folderpath+'/'+j
            shutil.copytree(tem_path,group_path+'/'+j)
    for test_slice in range(10):
        tem_list=[0,1,2,3,4,5,6,7,8,9]
        del tem_list[test_slice]
        val_slice=np.random.choice(tem_list,1,replace=False)[0]
        group_path_test = root_path + '/datasliced' + '/{}'.format(test_slice)
        group_path_val = root_path + '/datasliced' + '/{}'.format(val_slice)
        shutil.copytree(group_path_test,root_path+'/origin_data{}'.format(test_slice)+'/test_data')
        shutil.copytree(group_path_val, root_path + '/origin_data{}'.format(test_slice) + '/val_data')
        for train_slice in range(10):
            if train_slice!=test_slice and train_slice!=val_slice:
                for folder_name in os.listdir(root_path+'/datasliced'+'/{}'.format(train_slice)):
                    shutil.copytree(root_path+'/datasliced'+'/{}'.format(train_slice)+'/{}'.format(folder_name),root_path+'/origin_data{}'.format(test_slice) + '/train_data/{}'.format(folder_name))



if __name__=='__main__':
    getrandom('.','./result')