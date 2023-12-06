import pickle
import os
import pandas as pd
import additional_greedy as ag
import numpy as np
import utils as u
# 这是用于查看pkl文件的脚本
if __name__ == "__main__":
    # read all data file
    dir_path = "origin_data/train_data"
    for project_name in os.listdir(dir_path):
        data=pd.read_pickle(dir_path + "/" + project_name + "/data.pkl")
        print(project_name)
        # get codeNum,methodNum,testNum
        codeNum=data['codeNum']
        methNum=data['methodNum']
        testNum=data['testNum']
        print("codeNum:%d,methodNum:%d,testNum:%d"%(codeNum,methNum,testNum))
    # data = pd.read_pickle('origin_data/train_data/activemq-junit/time.pkl')
    # print(type(data['matrix']))
    # print(data)
    '''
    dir_path = "origin_data/val_data"
    for project_name in os.listdir(dir_path):
        with open(dir_path + "/" + project_name + "/weight.pkl", 'rb') as f:
            data = pickle.load(f)
        matrix = np.array(data['mutant_matrix'])
        with open(dir_path + "/" + project_name + "/time.pkl", 'rb') as f1:
            time = pickle.load(f1)
        s, val = ag.a_g(matrix.T, time)
        data['t_weight_list'] = list(val)
        with open(dir_path + "/" + project_name + "/weight.pkl", 'wb') as f:
            pickle.dump(data, f)
    dir_path = "origin_data/train_data"
    for project_name in os.listdir(dir_path):
        with open(dir_path + "/" + project_name + "/weight.pkl", 'rb') as f:
            data = pickle.load(f)
        matrix = np.array(data['mutant_matrix'])
        with open(dir_path + "/" + project_name + "/time.pkl", 'rb') as f1:
            time = pickle.load(f1)
        s, val = ag.a_g(matrix.T, time)
        data['t_weight_list'] = list(val)
        with open(dir_path + "/" + project_name + "/weight.pkl", 'wb') as f:
            pickle.dump(data, f)
    '''
    '''
    dir_path = "origin_data/test_data"
    for project_name in os.listdir(dir_path):
        with open(dir_path + "/" + project_name + "/weight.pkl", 'rb') as f:
            data = pickle.load(f)
        matrix = np.array(data['mutant_matrix'])
        with open(dir_path + "/" + project_name + "/time.pkl", 'rb') as f1:
            time = pickle.load(f1)
        s, val = ag.a_g(matrix.T, time)
        data['t_weight_list'] = list(val)
        with open(dir_path + "/" + project_name + "/weight.pkl", 'wb') as f:
            pickle.dump(data, f)
    '''
'''
    data = pd.read_pickle('origin_data/train_data/activemq-junit/weight.pkl')
    matrix = np.array(data['mutant_matrix'])
    time = pd.read_pickle('origin_data/train_data/activemq-junit/time.pkl')
    s, val = ag.a_g(matrix.T, time)
    s1, val1 = ag.a_g_withouttime(matrix.T)
    apfdc = u.gen_apfdc_value(matrix, s, time)
    # print(apfdc, u.gen_apfdc_value(matrix, s1, time))

    print(ag.get_apfdc(s1, matrix.T, time), ag.get_apfdc(s, matrix.T, time))
    file = open('origin_data/train_data/activemq-junit/weight.pkl', 'rb')
    m = pickle.load(file, encoding='utf-8')
    print(m['weight_list'], list(val1))
    print(list(val), data['t_weight_list'])

    t = pd.read_pickle('origin_data/val_data/sentinel-apache-dubbo-adapter/time.pkl')
    w = pd.read_pickle('origin_data/val_data/sentinel-apache-dubbo-adapter/weight.pkl')
    # print(t)
    # print(len(t), len(w['mutant_matrix'][0]))
'''

'''
    with open('origin_data/train_data/activemq-junit/time.pkl', 'rb') as f:
        data = pickle.load(f)
    matrix = np.array(data['mutant_matrix'])
    time = pd.read_pickle('origin_data/train_data/activemq-junit/time.pkl')
    s, val = ag.a_g(matrix.T, time)
    data['t_weight_list'] = list(val)
    with open('origin_data/test_data/activemq-junit/time.pkl', 'wb') as f:
        pickle.dump(data, f)
'''



