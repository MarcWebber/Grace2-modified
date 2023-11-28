from utils import get_apfdc
import os
import pickle
import numpy as np

if __name__ == '__main__':
    dir_path = "origin_data/test_data"
    apfdcs = []
    ori_apfdcs = []
    for project_name in os.listdir(dir_path):
        with open(dir_path + "/" + project_name + "/weight.pkl", 'rb') as f:
            data = pickle.load(f)
        enum_seq = sorted(enumerate(data['weight_list']), key=lambda x:x[1], reverse=True)
        ori_seq = [i[0] for i in enum_seq]
        matrix = np.array(data['mutant_matrix'])
        with open(dir_path + "/" + project_name + "/time.pkl", 'rb') as f1:
            time = pickle.load(f1)
        with open('data/test_sequene/19/' + project_name + '.pkl', 'rb') as f2:
            seq = pickle.load(f2)
        _, apfdc = get_apfdc(seq, matrix, time)
        _, ori_apfdc = get_apfdc(ori_seq, matrix, time)
        apfdcs.append(apfdc)
        ori_apfdcs.append(ori_apfdc)
    print(sum(apfdcs) / len(apfdcs))
    print(sum(ori_apfdcs) / len(ori_apfdcs))


